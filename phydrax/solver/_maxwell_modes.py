#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import ArraySpace, DenseLinearOperator, eigen as eigen_linalg, FailurePolicy
from ._maxwell_observers import ModeAmplitudeObserverPlan
from ._maxwell_sources import (
    AbstractMaxwellSourcePlan,
    MaxwellPairedCurrentSourcePlan,
    PreparedMaxwellSource,
)


class GuidedModeStatus(IntEnum):
    """Portable status for a fixed-frequency Maxwell polynomial solve."""

    SUCCESS = 0
    EIGENSOLVE_FAILURE = 1
    NONFINITE_OUTPUT = 2
    POWER_NORMALIZATION_FAILURE = 3
    ADJOINT_NORMALIZATION_FAILURE = 4
    DIVERGENCE_TOLERANCE_NOT_MET = 5
    BIORTHOGONALITY_TOLERANCE_NOT_MET = 6


class GuidedModeClassification(IntEnum):
    """Physical classification of one finite propagation constant."""

    INVALID = 0
    CUTOFF = 1
    PROPAGATING = 2
    EVANESCENT = 3
    LEAKY_OR_PML = 4


class GuidedModeDerivativeEvidence(StrictModule):
    """Evidence distinguishing simple-mode from invariant-subspace derivatives."""

    nearest_absolute_gaps: Array
    nearest_relative_gaps: Array
    eigenvalue_condition_estimates: Array
    isolated_mask: Array
    subspace_labels: Array
    cluster_multiplicities: Array
    finite_mask: Array
    gap_certified_mask: Array
    derivative_valid_mask: Array


class GuidedModeBetaDerivative(StrictModule):
    """Explicit first-order propagation-constant derivatives for simple modes."""

    values: Array
    valid_mask: Array
    evidence: GuidedModeDerivativeEvidence


class GuidedModeLaunch(StrictModule):
    """Power-normalized transverse traces for launching one solved mode."""

    electric_trace: Array
    magnetic_trace: Array
    amplitude: Array
    mode_id: str = eqx.field(static=True)


class FixedFrequencyGuidedModeResult(StrictModule):
    """Fixed-shape guided Maxwell modes and their numerical evidence."""

    angular_frequency: Array
    propagation_constants: Array
    right_coordinates: Array
    left_coordinates: Array
    right_electric_traces: Array
    right_magnetic_traces: Array
    left_electric_traces: Array
    left_magnetic_traces: Array
    polynomial_residuals: Array
    divergence_residuals: Array
    complex_powers: Array
    signed_powers: Array
    flux_matrix: Array
    biorthogonality_matrix: Array
    biorthogonality_error: Array
    classifications: Array
    derivative_evidence: GuidedModeDerivativeEvidence
    status: Array
    diagnostics: eigen_linalg.PolynomialEigenSolveDiagnostics
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(GuidedModeStatus.SUCCESS)

    def launch(
        self,
        mode: int,
        /,
        *,
        amplitude: ArrayLike = 1.0,
    ) -> GuidedModeLaunch:
        index = int(mode)
        if index < 0 or index >= len(self.mode_ids):
            raise IndexError("mode is outside the solved guided-mode range.")
        amplitude_ = jnp.asarray(amplitude, dtype=self.right_electric_traces.dtype)
        if amplitude_.ndim != 0:
            raise ValueError("amplitude must be one scalar.")
        amplitude_ = eqx.error_if(
            amplitude_, ~jnp.isfinite(amplitude_), "amplitude must be finite."
        )
        return GuidedModeLaunch(
            electric_trace=amplitude_ * self.right_electric_traces[:, index],
            magnetic_trace=amplitude_ * self.right_magnetic_traces[:, index],
            amplitude=amplitude_,
            mode_id=self.mode_ids[index],
        )


class FixedFrequencyGuidedModePlan(StrictModule, NonTrainableState):
    """Polynomial Maxwell pencil ``A₀ + β A₁ + β² A₂`` at fixed frequency.

    Trace maps are polynomial in ``β`` and are supplied independently for right
    and adjoint modes. This avoids silently identifying reciprocal, Hermitian,
    lossy, and PML adjoints. The power pairing maps magnetic traces to electric
    traces and fixes the right-mode normalization.
    """

    coefficients: tuple[Array, Array, Array]
    right_electric_trace_coefficients: tuple[Array, ...]
    right_magnetic_trace_coefficients: tuple[Array, ...]
    left_electric_trace_coefficients: tuple[Array, ...]
    left_magnetic_trace_coefficients: tuple[Array, ...]
    divergence_coefficients: tuple[Array, ...]
    power_pairing: Array
    angular_frequency: Array
    polynomial_policy: eigen_linalg.PolynomialEigenSolvePolicy
    mode_count: int = eqx.field(static=True)
    target_propagation_constant: complex = eqx.field(static=True)
    divergence_tolerance: float = eqx.field(static=True)
    biorthogonality_tolerance: float = eqx.field(static=True)
    classification_tolerance: float = eqx.field(static=True)
    isolation_absolute_tolerance: float = eqx.field(static=True)
    isolation_relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coefficient_0: ArrayLike,
        coefficient_1: ArrayLike,
        coefficient_2: ArrayLike,
        mode_count: int,
        /,
        *,
        angular_frequency: ArrayLike,
        right_electric_trace_coefficients: Sequence[ArrayLike],
        right_magnetic_trace_coefficients: Sequence[ArrayLike],
        left_electric_trace_coefficients: Sequence[ArrayLike],
        left_magnetic_trace_coefficients: Sequence[ArrayLike],
        divergence_coefficients: Sequence[ArrayLike],
        power_pairing: ArrayLike,
        target_propagation_constant: complex = 0.0,
        polynomial_policy: eigen_linalg.PolynomialEigenSolvePolicy | None = None,
        divergence_tolerance: float = 1e-8,
        biorthogonality_tolerance: float = 1e-8,
        classification_tolerance: float = 1e-9,
        isolation_absolute_tolerance: float = 1e-8,
        isolation_relative_tolerance: float = 1e-6,
        maximum_dofs: int = 4096,
    ):
        coefficient_values = tuple(
            np.asarray(value) for value in (coefficient_0, coefficient_1, coefficient_2)
        )
        dimension = int(coefficient_values[0].shape[0])
        if (
            any(
                value.ndim != 2 or value.shape != (dimension, dimension)
                for value in coefficient_values
            )
            or dimension < 1
        ):
            raise ValueError(
                "Guided-mode coefficients must be equally sized square matrices."
            )
        if np.linalg.matrix_rank(coefficient_values[2]) == 0:
            raise ValueError("coefficient_2 must be nonzero for a quadratic pencil.")
        if any(np.any(~np.isfinite(value)) for value in coefficient_values):
            raise ValueError("Guided-mode polynomial coefficients must be finite.")
        count = int(mode_count)
        if count < 1 or count > 2 * dimension:
            raise ValueError(
                "mode_count is outside the quadratic linearization dimension."
            )
        maximum = int(maximum_dofs)
        if maximum < 1 or dimension > maximum:
            raise ValueError("Guided-mode solve exceeds maximum_dofs.")
        frequency = np.asarray(angular_frequency)
        if (
            frequency.ndim != 0
            or not np.isrealobj(frequency)
            or not np.isfinite(frequency)
            or float(frequency) <= 0.0
        ):
            raise ValueError("angular_frequency must be one positive finite real scalar.")
        target = complex(target_propagation_constant)
        if not math.isfinite(target.real) or not math.isfinite(target.imag):
            raise ValueError("target_propagation_constant must be finite.")
        right_e = _guided_trace_coefficients(
            right_electric_trace_coefficients, dimension, "right electric"
        )
        right_h = _guided_trace_coefficients(
            right_magnetic_trace_coefficients, dimension, "right magnetic"
        )
        left_e = _guided_trace_coefficients(
            left_electric_trace_coefficients,
            dimension,
            "left electric",
            output_size=right_e[0].shape[0],
        )
        left_h = _guided_trace_coefficients(
            left_magnetic_trace_coefficients,
            dimension,
            "left magnetic",
            output_size=right_h[0].shape[0],
        )
        divergence = _guided_trace_coefficients(
            divergence_coefficients, dimension, "divergence"
        )
        pairing = np.asarray(power_pairing)
        if pairing.shape != (right_e[0].shape[0], right_h[0].shape[0]):
            raise ValueError("power_pairing must map magnetic traces to electric traces.")
        if np.any(~np.isfinite(pairing)):
            raise ValueError("power_pairing must be finite.")
        tolerances = tuple(
            float(value)
            for value in (
                divergence_tolerance,
                biorthogonality_tolerance,
                classification_tolerance,
                isolation_absolute_tolerance,
                isolation_relative_tolerance,
            )
        )
        if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("Guided-mode tolerances must be finite and non-negative.")
        selected_policy = (
            eigen_linalg.PolynomialEigenSolvePolicy(
                general=eigen_linalg.GeneralEigenSolvePolicy(
                    eigen_linalg.DenseSchurQZ(),
                    selection=eigen_linalg.GeneralEigenSelection.all(),
                    resources=eigen_linalg.GeneralEigenResourcePolicy(
                        max_dimension=2 * maximum
                    ),
                    failure=FailurePolicy("status"),
                ),
                eigenvalue_scale=max(1.0, abs(target)),
                relative_residual_tolerance=1e-8,
                absolute_residual_tolerance=1e-10,
            )
            if polynomial_policy is None
            else polynomial_policy
        )
        if not isinstance(selected_policy, eigen_linalg.PolynomialEigenSolvePolicy):
            raise TypeError(
                "polynomial_policy must be PolynomialEigenSolvePolicy or None."
            )
        available = selected_policy.general.selection.count
        if available is not None and available < count:
            raise ValueError(
                "polynomial_policy must retain at least mode_count eigenpairs."
            )
        self.coefficients = tuple(jnp.asarray(value) for value in coefficient_values)
        self.right_electric_trace_coefficients = tuple(
            jnp.asarray(value) for value in right_e
        )
        self.right_magnetic_trace_coefficients = tuple(
            jnp.asarray(value) for value in right_h
        )
        self.left_electric_trace_coefficients = tuple(
            jnp.asarray(value) for value in left_e
        )
        self.left_magnetic_trace_coefficients = tuple(
            jnp.asarray(value) for value in left_h
        )
        self.divergence_coefficients = tuple(jnp.asarray(value) for value in divergence)
        self.power_pairing = jnp.asarray(pairing)
        self.angular_frequency = jnp.asarray(frequency)
        self.polynomial_policy = selected_policy
        self.mode_count = count
        self.target_propagation_constant = target
        (
            self.divergence_tolerance,
            self.biorthogonality_tolerance,
            self.classification_tolerance,
            self.isolation_absolute_tolerance,
            self.isolation_relative_tolerance,
        ) = tolerances
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-frequency-guided-maxwell-mode-plan",
                "coefficients": array_tree_fingerprint(coefficient_values),
                "right_electric_traces": array_tree_fingerprint(right_e),
                "right_magnetic_traces": array_tree_fingerprint(right_h),
                "left_electric_traces": array_tree_fingerprint(left_e),
                "left_magnetic_traces": array_tree_fingerprint(left_h),
                "divergence": array_tree_fingerprint(divergence),
                "power_pairing": array_tree_fingerprint(pairing),
                "angular_frequency": float(frequency),
                "mode_count": count,
                "target_propagation_constant": [target.real, target.imag],
                "polynomial_policy": selected_policy.policy_id,
                "tolerances": tolerances,
            }
        )

    def prepare(self, /) -> "PreparedFixedFrequencyGuidedModes":
        return prepare_fixed_frequency_guided_modes(self)

    def solve(self, /) -> FixedFrequencyGuidedModeResult:
        return solve_fixed_frequency_guided_modes(self.prepare())


class PreparedFixedFrequencyGuidedModes(StrictModule):
    """Reusable fixed-shape numerical state for one guided Maxwell pencil."""

    plan: FixedFrequencyGuidedModePlan
    polynomial: eigen_linalg.PreparedPolynomialEigenSolve
    prepared_id: str = eqx.field(static=True)


def prepare_fixed_frequency_guided_modes(
    plan: FixedFrequencyGuidedModePlan,
    /,
) -> PreparedFixedFrequencyGuidedModes:
    if not isinstance(plan, FixedFrequencyGuidedModePlan):
        raise TypeError("plan must be a FixedFrequencyGuidedModePlan.")
    dtype = jnp.result_type(*(value.dtype for value in plan.coefficients), jnp.complex64)
    space = ArraySpace((plan.coefficients[0].shape[0],), dtype=dtype)
    operators = tuple(
        DenseLinearOperator(
            value.astype(dtype),
            source=space,
            target=space,
            operator_id=f"{plan.plan_id}:coefficient:{degree}",
        )
        for degree, value in enumerate(plan.coefficients)
    )
    problem = eigen_linalg.PolynomialEigenproblem(
        operators,
        problem_id=f"{plan.plan_id}:polynomial",
    )
    polynomial = eigen_linalg.prepare_polynomial_eigensolve(
        problem, plan.polynomial_policy
    )
    return PreparedFixedFrequencyGuidedModes(
        plan=plan,
        polynomial=polynomial,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-fixed-frequency-guided-maxwell-modes",
                "plan": plan.plan_id,
                "polynomial": polynomial.prepared_id,
            }
        ),
    )


def solve_fixed_frequency_guided_modes(
    prepared: PreparedFixedFrequencyGuidedModes,
    /,
) -> FixedFrequencyGuidedModeResult:
    if not isinstance(prepared, PreparedFixedFrequencyGuidedModes):
        raise TypeError("prepared must be PreparedFixedFrequencyGuidedModes.")
    plan = prepared.plan
    solved = eigen_linalg.polynomial_eigensolve(prepared.polynomial)
    full_beta = solved.eigenvalues
    spectrum_indices = jnp.argsort(jnp.abs(full_beta - plan.target_propagation_constant))[
        : plan.mode_count
    ]
    beta = full_beta[spectrum_indices]
    right_raw = jnp.asarray(solved.right_eigenvectors)[:, spectrum_indices]
    left_raw = jnp.asarray(solved.left_eigenvectors)[:, spectrum_indices]
    selected_finite = solved.diagnostics.finite_mask[spectrum_indices]
    right_e_raw = _evaluate_guided_trace(
        plan.right_electric_trace_coefficients, beta, right_raw
    )
    right_h_raw = _evaluate_guided_trace(
        plan.right_magnetic_trace_coefficients, beta, right_raw
    )
    raw_flux = 0.5 * (
        jnp.conj(jnp.swapaxes(right_e_raw, -1, -2)) @ plan.power_pairing @ right_h_raw
    )
    raw_powers = jnp.diag(raw_flux)
    real_power = jnp.abs(jnp.real(raw_powers))
    power_scale = jnp.where(
        real_power > plan.classification_tolerance,
        real_power,
        jnp.abs(raw_powers),
    )
    power_valid = jnp.isfinite(power_scale) & (
        power_scale > plan.classification_tolerance
    )
    right_scale = _safe_inverse_sqrt(power_scale, power_valid)
    right = right_raw * right_scale[None, :]
    coefficient_1 = plan.coefficients[1].astype(right.dtype)
    coefficient_2 = plan.coefficients[2].astype(right.dtype)
    derivative_images = (
        coefficient_1 @ right_raw + 2.0 * (coefficient_2 @ right_raw) * beta[None, :]
    )
    derivative_pairing_raw = jnp.sum(jnp.conj(left_raw) * derivative_images, axis=0)
    derivative_pairing_valid = jnp.isfinite(derivative_pairing_raw) & (
        jnp.abs(derivative_pairing_raw) > plan.biorthogonality_tolerance
    )
    safe_derivative_pairing = jnp.where(
        derivative_pairing_valid, derivative_pairing_raw, 1.0
    )
    left_scale = jnp.conj(1.0 / (right_scale * safe_derivative_pairing))
    left = left_raw * left_scale[None, :]
    right_e = _evaluate_guided_trace(plan.right_electric_trace_coefficients, beta, right)
    right_h = _evaluate_guided_trace(plan.right_magnetic_trace_coefficients, beta, right)
    left_e = _evaluate_guided_trace(plan.left_electric_trace_coefficients, beta, left)
    left_h = _evaluate_guided_trace(plan.left_magnetic_trace_coefficients, beta, left)
    divergence = _evaluate_guided_trace(plan.divergence_coefficients, beta, right)
    coordinate_norm = jnp.sqrt(jnp.sum(jnp.abs(right) ** 2, axis=0))
    divergence_residuals = jnp.sqrt(
        jnp.sum(jnp.abs(divergence) ** 2, axis=0)
    ) / jnp.maximum(coordinate_norm, jnp.finfo(coordinate_norm.dtype).tiny)
    flux_matrix = 0.5 * (
        jnp.conj(jnp.swapaxes(right_e, -1, -2)) @ plan.power_pairing @ right_h
    )
    complex_powers = jnp.diag(flux_matrix)
    signed_powers = jnp.sign(jnp.real(complex_powers))
    biorthogonality = jnp.conj(jnp.swapaxes(left, -1, -2)) @ coefficient_1 @ right + (
        beta[:, None] + beta[None, :]
    ) * (jnp.conj(jnp.swapaxes(left, -1, -2)) @ coefficient_2 @ right)
    identity = jnp.eye(plan.mode_count, dtype=biorthogonality.dtype)
    biorthogonality_error = jnp.max(jnp.abs(biorthogonality - identity))
    evidence = _guided_mode_derivative_evidence(
        beta,
        derivative_pairing_raw,
        selected_finite,
        full_beta,
        solved.diagnostics.finite_mask,
        spectrum_indices,
        plan,
    )
    classifications = _classify_guided_modes(beta, selected_finite, plan)
    finite_output = (
        jnp.all(jnp.isfinite(beta))
        & jnp.all(jnp.isfinite(right))
        & jnp.all(jnp.isfinite(left))
        & jnp.all(jnp.isfinite(right_e))
        & jnp.all(jnp.isfinite(right_h))
        & jnp.all(jnp.isfinite(left_e))
        & jnp.all(jnp.isfinite(left_h))
        & jnp.all(jnp.isfinite(divergence_residuals))
    )
    selected_converged = jnp.all(
        solved.diagnostics.converged_mask[spectrum_indices] & selected_finite
    )
    status = jnp.where(
        ~selected_converged,
        int(GuidedModeStatus.EIGENSOLVE_FAILURE),
        jnp.where(
            ~finite_output,
            int(GuidedModeStatus.NONFINITE_OUTPUT),
            jnp.where(
                ~jnp.all(power_valid),
                int(GuidedModeStatus.POWER_NORMALIZATION_FAILURE),
                jnp.where(
                    ~jnp.all(derivative_pairing_valid),
                    int(GuidedModeStatus.ADJOINT_NORMALIZATION_FAILURE),
                    jnp.where(
                        jnp.any(divergence_residuals > plan.divergence_tolerance),
                        int(GuidedModeStatus.DIVERGENCE_TOLERANCE_NOT_MET),
                        jnp.where(
                            biorthogonality_error > plan.biorthogonality_tolerance,
                            int(GuidedModeStatus.BIORTHOGONALITY_TOLERANCE_NOT_MET),
                            int(GuidedModeStatus.SUCCESS),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    mode_ids = tuple(f"{plan.plan_id}:mode:{index}" for index in range(plan.mode_count))
    return FixedFrequencyGuidedModeResult(
        angular_frequency=plan.angular_frequency,
        propagation_constants=beta,
        right_coordinates=right,
        left_coordinates=left,
        right_electric_traces=right_e,
        right_magnetic_traces=right_h,
        left_electric_traces=left_e,
        left_magnetic_traces=left_h,
        polynomial_residuals=solved.diagnostics.original_residual_norms[spectrum_indices],
        divergence_residuals=divergence_residuals,
        complex_powers=complex_powers,
        signed_powers=signed_powers,
        flux_matrix=flux_matrix,
        biorthogonality_matrix=biorthogonality,
        biorthogonality_error=biorthogonality_error,
        classifications=classifications,
        derivative_evidence=evidence,
        status=status,
        diagnostics=solved.diagnostics,
        mode_ids=mode_ids,
        prepared_id=prepared.prepared_id,
        result_id=canonical_fingerprint(
            {
                "kind": "fixed-frequency-guided-maxwell-mode-result",
                "prepared": prepared.prepared_id,
                "polynomial_plan": solved.provenance.plan_id,
            }
        ),
    )


def guided_mode_beta_derivative(
    prepared: PreparedFixedFrequencyGuidedModes,
    result: FixedFrequencyGuidedModeResult,
    coefficient_tangents: Sequence[ArrayLike],
    /,
) -> GuidedModeBetaDerivative:
    """Differentiate isolated propagation constants with the polynomial adjoint."""
    if not isinstance(prepared, PreparedFixedFrequencyGuidedModes):
        raise TypeError("prepared must be PreparedFixedFrequencyGuidedModes.")
    if not isinstance(result, FixedFrequencyGuidedModeResult):
        raise TypeError("result must be a FixedFrequencyGuidedModeResult.")
    if result.prepared_id != prepared.prepared_id:
        raise ValueError("result was not produced by prepared.")
    tangents = tuple(jnp.asarray(value) for value in coefficient_tangents)
    expected_shape = prepared.plan.coefficients[0].shape
    if len(tangents) != 3 or any(value.shape != expected_shape for value in tangents):
        raise ValueError("coefficient_tangents must contain three pencil-sized matrices.")
    beta = result.propagation_constants
    tangent_image = (
        tangents[0] @ result.right_coordinates
        + (tangents[1] @ result.right_coordinates) * beta[None, :]
        + (tangents[2] @ result.right_coordinates) * beta[None, :] ** 2
    )
    numerator = jnp.sum(
        jnp.conj(result.left_coordinates) * tangent_image,
        axis=0,
    )
    values = -numerator
    valid = (
        result.derivative_evidence.derivative_valid_mask
        & jnp.isfinite(values)
        & result.successful
    )
    values = jnp.where(valid, values, jnp.nan + 0j)
    return GuidedModeBetaDerivative(values, valid, result.derivative_evidence)


def _guided_trace_coefficients(
    values: Sequence[ArrayLike],
    dimension: int,
    name: str,
    /,
    *,
    output_size: int | None = None,
) -> tuple[np.ndarray, ...]:
    coefficients = tuple(np.asarray(value) for value in values)
    if not coefficients:
        raise ValueError(f"{name} trace coefficients must be non-empty.")
    rows = int(coefficients[0].shape[0])
    if rows < 1 or any(
        value.ndim != 2 or value.shape != (rows, dimension) for value in coefficients
    ):
        raise ValueError(f"{name} trace coefficients must share shape (trace, dof).")
    if output_size is not None and rows != output_size:
        raise ValueError(f"{name} trace output size does not match its right trace.")
    if any(np.any(~np.isfinite(value)) for value in coefficients):
        raise ValueError(f"{name} trace coefficients must be finite.")
    return coefficients


def _evaluate_guided_trace(
    coefficients: tuple[Array, ...],
    beta: Array,
    coordinates: Array,
    /,
) -> Array:
    dtype = jnp.result_type(
        coordinates.dtype, beta.dtype, *(value.dtype for value in coefficients)
    )
    result = jnp.zeros((coefficients[0].shape[0], coordinates.shape[1]), dtype=dtype)
    for degree, coefficient in enumerate(coefficients):
        result = result + (coefficient @ coordinates) * beta[None, :] ** degree
    return result


def _safe_inverse_sqrt(value: Array, valid: Array, /) -> Array:
    safe = jnp.where(valid, value, 1.0)
    return jnp.where(valid, 1.0 / jnp.sqrt(safe), 1.0)


def _guided_mode_derivative_evidence(
    beta,
    pairing,
    finite,
    full_beta,
    full_finite,
    spectrum_indices,
    plan,
):
    count = plan.mode_count
    distances = jnp.abs(beta[:, None] - full_beta[None, :])
    same_mode = (
        jnp.arange(full_beta.size, dtype=jnp.int32)[None, :] == spectrum_indices[:, None]
    )
    distances = jnp.where(same_mode | ~full_finite[None, :], jnp.inf, distances)
    nearest = jnp.min(distances, axis=1)
    scale = jnp.maximum(1.0, jnp.abs(beta))
    relative = nearest / scale
    threshold = plan.isolation_absolute_tolerance + (
        plan.isolation_relative_tolerance
        * jnp.maximum(jnp.abs(beta[:, None]), jnp.abs(full_beta[None, :]))
    )
    close_to_spectrum = (distances <= threshold) & full_finite[None, :] & ~same_mode
    isolated = ~jnp.any(close_to_spectrum, axis=1)
    multiplicities = 1 + jnp.sum(close_to_spectrum, axis=1, dtype=jnp.int32)
    selected_distances = jnp.abs(beta[:, None] - beta[None, :])
    selected_threshold = plan.isolation_absolute_tolerance + (
        plan.isolation_relative_tolerance
        * jnp.maximum(jnp.abs(beta[:, None]), jnp.abs(beta[None, :]))
    )
    selected_close = (selected_distances <= selected_threshold) & ~jnp.eye(
        count, dtype=bool
    )
    indices = jnp.arange(count, dtype=jnp.int32)
    labels = jnp.min(
        jnp.where(
            selected_close | jnp.eye(count, dtype=bool),
            indices[None, :],
            count,
        ),
        axis=1,
    )
    pairing_magnitude = jnp.abs(pairing)
    condition = jnp.where(
        pairing_magnitude > 0.0,
        1.0 / pairing_magnitude,
        jnp.inf,
    )
    finite_mask = finite & jnp.isfinite(beta) & jnp.isfinite(condition)
    gap_certified = jnp.full(
        (count,),
        plan.polynomial_policy.general.selection.kind == "all",
        dtype=bool,
    )
    valid = finite_mask & isolated & gap_certified
    return GuidedModeDerivativeEvidence(
        nearest_absolute_gaps=nearest,
        nearest_relative_gaps=relative,
        eigenvalue_condition_estimates=condition,
        isolated_mask=isolated,
        subspace_labels=labels,
        finite_mask=finite_mask,
        cluster_multiplicities=multiplicities,
        gap_certified_mask=gap_certified,
        derivative_valid_mask=valid,
    )


def _classify_guided_modes(beta, finite, plan):
    real = jnp.abs(jnp.real(beta))
    imaginary = jnp.abs(jnp.imag(beta))
    scale = jnp.maximum(1.0, jnp.abs(beta))
    tolerance = plan.classification_tolerance * scale
    cutoff = (real <= tolerance) & (imaginary <= tolerance)
    propagating = (real > tolerance) & (imaginary <= tolerance)
    evanescent = (real <= tolerance) & (imaginary > tolerance)
    return jnp.where(
        ~finite,
        int(GuidedModeClassification.INVALID),
        jnp.where(
            cutoff,
            int(GuidedModeClassification.CUTOFF),
            jnp.where(
                propagating,
                int(GuidedModeClassification.PROPAGATING),
                jnp.where(
                    evanescent,
                    int(GuidedModeClassification.EVANESCENT),
                    int(GuidedModeClassification.LEAKY_OR_PML),
                ),
            ),
        ),
    ).astype(jnp.int32)


class MaxwellHuygensSourcePlan(AbstractMaxwellSourcePlan, NonTrainableState):
    """Orientation-certified paired equivalent currents on one discrete surface."""

    paired: MaxwellPairedCurrentSourcePlan
    direction: int = eqx.field(static=True)
    signed_power: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        electric_indices: ArrayLike,
        magnetic_trace: ArrayLike,
        magnetic_indices: ArrayLike,
        electric_trace: ArrayLike,
        /,
        *,
        signed_power: float,
        direction: int = 1,
        angular_frequency: ArrayLike = 0.0,
        phase: ArrayLike = 0.0,
        amplitude: ArrayLike = 1.0,
        control_key: str | None = None,
        magnetic_closedness_preserving: bool = False,
    ):
        if direction not in (-1, 1):
            raise ValueError("Huygens launch direction must be -1 or +1.")
        power = float(signed_power)
        if not np.isfinite(power) or power == 0.0:
            raise ValueError("Huygens launch requires finite nonzero signed power.")
        identifier = canonical_fingerprint(
            {
                "kind": "maxwell-huygens-source-plan",
                "electric_indices": array_tree_fingerprint(electric_indices),
                "magnetic_indices": array_tree_fingerprint(magnetic_indices),
                "direction": direction,
                "signed_power": power,
            }
        )
        self.paired = MaxwellPairedCurrentSourcePlan(
            electric_indices,
            direction * jnp.asarray(magnetic_trace),
            magnetic_indices,
            -direction * jnp.asarray(electric_trace),
            angular_frequency=angular_frequency,
            phase=phase,
            amplitude=amplitude / np.sqrt(abs(power)),
            control_key=control_key,
            magnetic_closedness_preserving=magnetic_closedness_preserving,
            source_id=identifier,
        )
        self.direction, self.signed_power, self.source_id = (
            int(direction),
            power,
            identifier,
        )

    def prepare(self, bridge, layout, /) -> PreparedMaxwellSource:
        return self.paired.prepare(bridge, layout)


class MaxwellModePortResponse(StrictModule):
    incident: Array
    reflected: Array
    transmitted: Array
    reflection: Array
    transmission: Array
    power_balance: Array


class MaxwellModePortPlan(StrictModule):
    source: MaxwellHuygensSourcePlan
    observer: ModeAmplitudeObserverPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: MaxwellHuygensSourcePlan,
        observer: ModeAmplitudeObserverPlan,
        /,
    ):
        if not isinstance(source, MaxwellHuygensSourcePlan) or not isinstance(
            observer, ModeAmplitudeObserverPlan
        ):
            raise TypeError("Mode port requires a paired mode source and modal observer.")
        self.source, self.observer = source, observer
        self.plan_id = canonical_fingerprint(
            {
                "kind": "maxwell-mode-port-plan",
                "source": source.source_id,
                "observer": observer.plan_id,
            }
        )

    def response(
        self,
        incident: ArrayLike,
        reflected: ArrayLike,
        transmitted: ArrayLike,
        /,
    ) -> MaxwellModePortResponse:
        incident_, reflected_, transmitted_ = (
            jnp.asarray(value) for value in (incident, reflected, transmitted)
        )
        if incident_.shape != reflected_.shape or incident_.shape != transmitted_.shape:
            raise ValueError("Mode-port amplitude arrays must have matching shapes.")
        valid = jnp.abs(incident_) > 0.0
        safe = jnp.where(valid, incident_, 1.0)
        reflection = jnp.where(valid, reflected_ / safe, jnp.nan)
        transmission = jnp.where(valid, transmitted_ / safe, jnp.nan)
        return MaxwellModePortResponse(
            incident_,
            reflected_,
            transmitted_,
            reflection,
            transmission,
            jnp.abs(reflection) ** 2 + jnp.abs(transmission) ** 2,
        )


class MaxwellModeDecomposition(StrictModule, NonTrainableState):
    modes: Array
    mass: Array
    decomposition_id: str = eqx.field(static=True)

    def __init__(self, modes: ArrayLike, mass: ArrayLike, /):
        modes_ = jnp.asarray(modes)
        mass_ = jnp.asarray(mass)
        if modes_.ndim != 2 or mass_.shape != (modes_.shape[0], modes_.shape[0]):
            raise ValueError("Mode decomposition shapes are incompatible.")
        gram = jnp.conj(modes_.T) @ mass_ @ modes_
        if not np.allclose(np.asarray(gram), np.eye(modes_.shape[1]), atol=1e-8):
            raise ValueError("Mode basis must be mass-orthonormal.")
        self.modes = modes_
        self.mass = mass_
        self.decomposition_id = canonical_fingerprint(
            {
                "kind": "maxwell-mode-decomposition",
                "modes": array_tree_fingerprint(modes_),
                "mass": array_tree_fingerprint(mass_),
            }
        )

    def amplitudes(self, field: ArrayLike, /) -> Array:
        value = jnp.asarray(field)
        if value.shape != self.modes.shape[:1]:
            raise ValueError("Field shape does not match mode basis.")
        return jnp.conj(self.modes.T) @ (self.mass @ value)


class MaxwellNearToFarPlan(StrictModule, NonTrainableState):
    """Homogeneous-exterior Huygens surface far-field transform."""

    positions: Array
    normals: Array
    weights: Array
    directions: Array
    wavenumbers: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        positions: ArrayLike,
        normals: ArrayLike,
        weights: ArrayLike,
        directions: ArrayLike,
        wavenumbers: ArrayLike,
        /,
    ):
        positions_ = jnp.asarray(positions, dtype=float)
        normals_ = jnp.asarray(normals, dtype=float)
        weights_ = jnp.asarray(weights, dtype=float)
        directions_ = jnp.asarray(directions, dtype=float)
        wavenumbers_ = jnp.asarray(wavenumbers, dtype=float)
        if positions_.ndim != 2 or positions_.shape[1] != 3:
            raise ValueError("Near-to-far positions must have shape (surface, 3).")
        if normals_.shape != positions_.shape or weights_.shape != positions_.shape[:1]:
            raise ValueError("Near-to-far surface arrays are incompatible.")
        if directions_.ndim != 2 or directions_.shape[1] != 3:
            raise ValueError("Far-field directions must have shape (directions, 3).")
        if wavenumbers_.ndim != 1 or wavenumbers_.size == 0:
            raise ValueError("wavenumbers must be a nonempty vector.")
        normal_norm = jnp.linalg.norm(normals_, axis=1)
        direction_norm = jnp.linalg.norm(directions_, axis=1)
        normals_ = normals_ / normal_norm[:, None]
        directions_ = directions_ / direction_norm[:, None]
        invalid = (
            jnp.any(~jnp.isfinite(positions_))
            | jnp.any(~jnp.isfinite(normals_))
            | jnp.any(~jnp.isfinite(weights_))
            | jnp.any(~jnp.isfinite(directions_))
            | jnp.any(~jnp.isfinite(wavenumbers_))
            | jnp.any(normal_norm <= 0.0)
            | jnp.any(direction_norm <= 0.0)
            | jnp.any(weights_ <= 0.0)
            | jnp.any(wavenumbers_ < 0.0)
        )
        positions_ = eqx.error_if(
            positions_, invalid, "Near-to-far geometry/frequencies are invalid."
        )
        self.positions = positions_
        self.normals = normals_
        self.weights = weights_
        self.directions = directions_
        self.wavenumbers = wavenumbers_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "maxwell-near-to-far",
                "positions": array_tree_fingerprint(positions_),
                "directions": array_tree_fingerprint(directions_),
                "wavenumbers": array_tree_fingerprint(wavenumbers_),
            }
        )

    def transform(self, electric: ArrayLike, magnetic: ArrayLike, /) -> Array:
        electric_ = jnp.asarray(electric)
        magnetic_ = jnp.asarray(magnetic)
        if (
            electric_.shape != self.positions.shape
            or magnetic_.shape != self.positions.shape
        ):
            raise ValueError("Near-to-far fields must have shape (surface, 3).")
        electric_current = jnp.cross(self.normals, magnetic_)
        magnetic_current = -jnp.cross(self.normals, electric_)
        phase_argument = self.directions @ self.positions.T
        phase = jnp.exp(
            -1j * self.wavenumbers[:, None, None] * phase_argument[None, :, :]
        )
        projected_electric = electric_current[None, None, :, :]
        projected_magnetic = jnp.cross(
            self.directions[None, :, None, :],
            magnetic_current[None, None, :, :],
        )
        integrand = projected_electric + projected_magnetic
        return jnp.sum(
            phase[..., None] * self.weights[None, None, :, None] * integrand,
            axis=2,
        )


__all__ = [
    "FixedFrequencyGuidedModePlan",
    "FixedFrequencyGuidedModeResult",
    "GuidedModeBetaDerivative",
    "GuidedModeClassification",
    "GuidedModeDerivativeEvidence",
    "GuidedModeLaunch",
    "GuidedModeStatus",
    "MaxwellHuygensSourcePlan",
    "MaxwellModeDecomposition",
    "MaxwellModePortPlan",
    "MaxwellModePortResponse",
    "MaxwellNearToFarPlan",
    "PreparedFixedFrequencyGuidedModes",
    "guided_mode_beta_derivative",
    "prepare_fixed_frequency_guided_modes",
    "solve_fixed_frequency_guided_modes",
]
