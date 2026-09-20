#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...continuation import ParameterContinuationProblem
from ...linalg import ArraySpace, DenseLinearOperator, svd as svd_api
from ...nonlinear import (
    AbstractNonlinearMethod,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
    root,
    root_solution_jvp,
    SensitivityEvidence,
    SensitivityPolicy,
)
from ._perturbation import SeparatedMode
from ._radial_perturbation import (
    evaluate_kerr_teukolsky_radial,
    evaluate_schwarzschild_radial,
    KerrTeukolskyRadialPlan,
    KerrTeukolskyRadialResult,
    SchwarzschildRadialPlan,
    SchwarzschildRadialResult,
)
from ._spheroidal import (
    solve_spheroidal_angular,
    SpheroidalAngularPlan,
    SpheroidalAngularResult,
)


class QnmStatus(IntEnum):
    """Terminal numerical and physical status for one complex QNM root."""

    SUCCESS = 0
    NONFINITE = 1
    NONLINEAR_NOT_CONVERGED = 2
    CONTINUED_FRACTION_DEPTH_UNRESOLVED = 3
    ANGULAR_RESOLUTION_UNRESOLVED = 4
    RADIAL_RESOLUTION_UNRESOLVED = 5
    NONDECAYING_MODE = 6
    NONSIMPLE_ROOT = 7


class QnmDerivativeStatus(IntEnum):
    """Validity of the branch-local implicit QNM derivative."""

    VALID = 0
    PRIMAL_NOT_CONVERGED = 1
    DEPTH_UNRESOLVED = 2
    ANGULAR_BRANCH_UNRESOLVED = 3
    RADIAL_RESOLUTION_UNRESOLVED = 4
    RADIAL_DERIVATIVE_UNRESOLVED = 5
    NONSIMPLE_ROOT = 6
    NONFINITE = 7
    SENSITIVITY_SOLVE_FAILED = 8


class BoundedContinuedFractionPlan(StrictModule, NonTrainableState):
    """Static truncation and inversion policy for one three-term recurrence."""

    comparison_depth: int = eqx.field(static=True)
    maximum_depth: int = eqx.field(static=True)
    inversion_index: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        comparison_depth: int,
        maximum_depth: int,
        inversion_index: int,
        absolute_tolerance: float,
        relative_tolerance: float,
        /,
    ):
        depths = (comparison_depth, maximum_depth, inversion_index)
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) for value in depths
        ):
            raise TypeError(
                "Continued-fraction depths and inversion index must be integers."
            )
        comparison = int(comparison_depth)
        maximum = int(maximum_depth)
        inversion = int(inversion_index)
        if inversion < 0 or comparison <= inversion or maximum <= comparison:
            raise ValueError(
                "Continued-fraction depths must satisfy inversion < comparison < maximum."
            )
        tolerances = float(absolute_tolerance), float(relative_tolerance)
        if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError(
                "Continued-fraction tolerances must be finite and non-negative."
            )
        if not any(value > 0.0 for value in tolerances):
            raise ValueError(
                "At least one continued-fraction tolerance must be positive."
            )
        self.comparison_depth = comparison
        self.maximum_depth = maximum
        self.inversion_index = inversion
        self.absolute_tolerance, self.relative_tolerance = tolerances
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bounded-three-term-continued-fraction",
                "comparison_depth": comparison,
                "maximum_depth": maximum,
                "inversion_index": inversion,
                "absolute_tolerance": tolerances[0],
                "relative_tolerance": tolerances[1],
            }
        )


class ContinuedFractionDepthEvidence(StrictModule):
    """Fine/coarse truncation values and an explicit depth-resolution decision."""

    value: Array
    comparison_value: Array
    truncation_error: Array
    threshold: Array
    finite: Array
    resolved: Array
    comparison_depth: Array
    maximum_depth: Array
    plan_id: str = eqx.field(static=True)


class QnmReferenceMode(StrictModule, NonTrainableState):
    """Explicit dimensionless reference root used only when qualification is requested."""

    dimensionless_spin: Array
    angular_frequency: Array
    separation_constant: Array
    frequency_tolerance: float = eqx.field(static=True)
    separation_tolerance: float = eqx.field(static=True)
    mode_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: SeparatedMode,
        dimensionless_spin: ArrayLike,
        angular_frequency: ArrayLike,
        separation_constant: ArrayLike,
        frequency_tolerance: float,
        separation_tolerance: float,
        /,
        *,
        source_id: str,
    ):
        if not isinstance(mode, SeparatedMode):
            raise TypeError("mode must be a SeparatedMode.")
        spin = _real_scalar(dimensionless_spin, "dimensionless_spin")
        frequency = _complex_scalar(angular_frequency, "angular_frequency")
        separation = _complex_scalar(separation_constant, "separation_constant")
        tolerances = float(frequency_tolerance), float(separation_tolerance)
        if any(not math.isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError("Reference tolerances must be finite and positive.")
        source = str(source_id)
        if not source:
            raise ValueError("source_id must be non-empty.")
        if not bool(
            jnp.isfinite(spin)
            & jnp.isfinite(jnp.real(frequency))
            & jnp.isfinite(jnp.imag(frequency))
            & jnp.isfinite(jnp.real(separation))
            & jnp.isfinite(jnp.imag(separation))
        ):
            raise ValueError("Reference spin, frequency, and separation must be finite.")
        self.dimensionless_spin = spin
        self.angular_frequency = frequency
        self.separation_constant = separation
        self.frequency_tolerance, self.separation_tolerance = tolerances
        self.mode_id = mode.mode_id
        self.source_id = source
        self.reference_id = canonical_fingerprint(
            {
                "kind": "qnm-reference-mode",
                "mode": mode.mode_id,
                "dimensionless_spin": float(spin),
                "angular_frequency": (
                    float(jnp.real(frequency)),
                    float(jnp.imag(frequency)),
                ),
                "separation_constant": (
                    float(jnp.real(separation)),
                    float(jnp.imag(separation)),
                ),
                "frequency_tolerance": tolerances[0],
                "separation_tolerance": tolerances[1],
                "source": source,
            }
        )


class QnmSolvePlan(StrictModule, NonTrainableState):
    """Coupled angular/radial Leaver root with delegated numerical policies.

    Frequencies are reported as dimensionless ``M omega``.  The recurrence uses
    ``2M=1`` internally, while its spheroidicity is the invariant ``a omega``.
    The nonlinear method, termination, sensitivity solve, root seed, bounded
    depths, and branch identity are all explicit; there is no fallback seed or
    hidden root algorithm.
    """

    mode: SeparatedMode
    angular_plan: SpheroidalAngularPlan
    radial_plan: SchwarzschildRadialPlan | KerrTeukolskyRadialPlan
    dimensionless_spin: Array
    nonlinear_method: AbstractNonlinearMethod
    termination: NonlinearTermination
    sensitivity: SensitivityPolicy
    angular_fraction: BoundedContinuedFractionPlan
    radial_fraction: BoundedContinuedFractionPlan
    angular_resolution_tolerance: float = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    continuation_lower: float = eqx.field(static=True)
    continuation_upper: float = eqx.field(static=True)
    radial_source_id: str = eqx.field(static=True)
    branch_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: SeparatedMode,
        angular_plan: SpheroidalAngularPlan,
        radial_plan: SchwarzschildRadialPlan | KerrTeukolskyRadialPlan,
        dimensionless_spin: ArrayLike,
        nonlinear_method: AbstractNonlinearMethod,
        termination: NonlinearTermination,
        sensitivity: SensitivityPolicy,
        angular_fraction: BoundedContinuedFractionPlan,
        radial_fraction: BoundedContinuedFractionPlan,
        angular_resolution_tolerance: float,
        condition_limit: float,
        continuation_bounds: tuple[float, float],
        /,
        *,
        branch_id: str,
    ):
        if not isinstance(mode, SeparatedMode):
            raise TypeError("mode must be a SeparatedMode.")
        if mode.family != "qnm":
            raise ValueError("QnmSolvePlan requires a mode with family='qnm'.")
        if mode.spin_weight not in (-2, -1, 0):
            raise ValueError(
                "The Leaver radial recurrence supports spin_weight -2, -1, or 0."
            )
        if not isinstance(angular_plan, SpheroidalAngularPlan):
            raise TypeError("angular_plan must be a SpheroidalAngularPlan.")
        if angular_plan.mode.mode_id != mode.mode_id:
            raise ValueError("Angular plan and QNM mode identities do not match.")
        if not isinstance(
            radial_plan, (SchwarzschildRadialPlan, KerrTeukolskyRadialPlan)
        ):
            raise TypeError(
                "radial_plan must be a SchwarzschildRadialPlan or KerrTeukolskyRadialPlan."
            )
        if radial_plan.mode.mode_id != mode.mode_id:
            raise ValueError("Radial plan and QNM mode identities do not match.")
        if not isinstance(nonlinear_method, AbstractNonlinearMethod):
            raise TypeError("nonlinear_method must be an AbstractNonlinearMethod.")
        if not isinstance(termination, NonlinearTermination):
            raise TypeError("termination must be a NonlinearTermination.")
        if not isinstance(sensitivity, SensitivityPolicy):
            raise TypeError("sensitivity must be a SensitivityPolicy.")
        if sensitivity.mode != "implicit-forward":
            raise ValueError("QNM spin derivatives require implicit-forward sensitivity.")
        for value, name in (
            (angular_fraction, "angular_fraction"),
            (radial_fraction, "radial_fraction"),
        ):
            if not isinstance(value, BoundedContinuedFractionPlan):
                raise TypeError(f"{name} must be a BoundedContinuedFractionPlan.")
        angular_inversion = mode.ell - max(abs(mode.m), abs(mode.spin_weight))
        if angular_fraction.inversion_index != angular_inversion:
            raise ValueError(
                "Angular continued-fraction inversion must select the declared ell branch."
            )
        if radial_fraction.inversion_index != mode.overtone:
            raise ValueError(
                "Radial continued-fraction inversion must select the declared overtone."
            )
        spin = _real_scalar(dimensionless_spin, "dimensionless_spin")
        if not bool(jnp.isfinite(spin)) or abs(float(spin)) >= 1.0:
            raise ValueError(
                "dimensionless_spin must be finite and strictly subextremal."
            )
        if isinstance(radial_plan, SchwarzschildRadialPlan):
            if float(spin) != 0.0:
                raise ValueError(
                    "A Schwarzschild radial plan requires dimensionless_spin=0."
                )
        else:
            radial_spin = float(radial_plan.spin / radial_plan.mass)
            if not math.isclose(
                radial_spin,
                float(spin),
                rel_tol=64.0 * np.finfo(np.float64).eps,
                abs_tol=64.0 * np.finfo(np.float64).eps,
            ):
                raise ValueError(
                    "Kerr radial spin/mass does not match dimensionless_spin."
                )
        resolution_tolerance = float(angular_resolution_tolerance)
        condition = float(condition_limit)
        if (
            not math.isfinite(resolution_tolerance)
            or resolution_tolerance <= 0.0
            or not math.isfinite(condition)
            or condition <= 1.0
        ):
            raise ValueError(
                "Angular resolution tolerance must be positive and condition_limit > 1."
            )
        if len(continuation_bounds) != 2:
            raise ValueError("continuation_bounds must contain lower and upper spin.")
        lower, upper = (float(value) for value in continuation_bounds)
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or not -1.0 < lower <= float(spin) <= upper < 1.0
        ):
            raise ValueError(
                "Continuation bounds must be finite, ordered, subextremal, and contain spin."
            )
        radial_source = radial_plan.plan_id
        branch = str(branch_id)
        if not branch:
            raise ValueError("branch_id must be non-empty.")
        self.mode = mode
        self.angular_plan = angular_plan
        self.radial_plan = radial_plan
        self.dimensionless_spin = spin
        self.nonlinear_method = nonlinear_method
        self.termination = termination
        self.sensitivity = sensitivity
        self.angular_fraction = angular_fraction
        self.radial_fraction = radial_fraction
        self.angular_resolution_tolerance = resolution_tolerance
        self.condition_limit = condition
        self.continuation_lower, self.continuation_upper = lower, upper
        self.radial_source_id = radial_source
        self.branch_id = branch
        self.plan_id = canonical_fingerprint(
            {
                "kind": "coupled-leaver-qnm-solve-plan",
                "mode": mode.mode_id,
                "angular_plan": angular_plan.plan_id,
                "dimensionless_spin": float(spin),
                "nonlinear_method": nonlinear_method.method_id,
                "termination": {
                    "absolute_residual": termination.absolute_residual,
                    "relative_residual": termination.relative_residual,
                    "maximum_residual": termination.maximum_residual,
                    "absolute_step": termination.absolute_step,
                    "relative_step": termination.relative_step,
                    "maximum_steps": termination.maximum_steps,
                    "maximum_evaluations": termination.maximum_evaluations,
                    "maximum_linear_iterations": termination.maximum_linear_iterations,
                    "divergence_factor": termination.divergence_factor,
                },
                "sensitivity": {
                    "mode": sensitivity.mode,
                    "iterations": sensitivity.iterations,
                    "truncation": sensitivity.truncation,
                    "condition_limit": sensitivity.condition_limit,
                    "perturbation": sensitivity.perturbation,
                    "linear_method": sensitivity.linear.method.name,
                    "linear_relative_tolerance": sensitivity.linear.tolerance.relative,
                    "linear_absolute_tolerance": sensitivity.linear.tolerance.absolute,
                    "linear_maximum_steps": sensitivity.linear.tolerance.max_steps,
                    "precision": sensitivity.precision.policy_id,
                },
                "angular_fraction": angular_fraction.plan_id,
                "radial_fraction": radial_fraction.plan_id,
                "angular_resolution_tolerance": resolution_tolerance,
                "condition_limit": condition,
                "continuation_bounds": (lower, upper),
                "radial_source": radial_plan.plan_id,
                "branch": branch,
            }
        )


class QnmResult(StrictModule):
    """One coupled complex QNM root with numerical, physical, and branch evidence."""

    angular_frequency: Array
    separation_constant: Array
    residuals: Array
    angular_residual: Array
    radial_residual: Array
    residual_norm: Array
    angular_depth_evidence: ContinuedFractionDepthEvidence
    radial_depth_evidence: ContinuedFractionDepthEvidence
    angular_resolution: SpheroidalAngularResult
    angular_resolution_error: Array
    radial_resolution: SchwarzschildRadialResult | KerrTeukolskyRadialResult
    radial_resolution_error: Array
    root_condition: Array
    minimum_singular_value: Array
    frequency_spin_derivative: Array
    separation_spin_derivative: Array
    sensitivity_evidence: SensitivityEvidence
    nonlinear_result: NonlinearResult
    reference_frequency_error: Array
    reference_separation_error: Array
    continuation_coordinate: Array
    continuation_active: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    status: Array
    derivative_status: Array
    mode_id: str = eqx.field(static=True)
    radial_source_id: str = eqx.field(static=True)
    qualification_source_id: str = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)
    branch_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        """Whether this is a finite converged physically admissible QNM root."""

        return self.finite & self.converged & self.physically_valid


_SCHWARZSCHILD_REFERENCE_FREQUENCIES: dict[tuple[int, int, int], complex] = {
    (0, 0, 0): 0.110454939080173 - 0.104895717086880j,
    (0, 1, 0): 0.292936133267285 - 0.097659988913578j,
    (0, 2, 0): 0.483643872210713 - 0.096758775978287j,
    (-1, 1, 0): 0.248263264178109 - 0.092487717952942j,
    (-2, 2, 0): 0.373671684418042 - 0.088962315688936j,
    (-2, 2, 1): 0.346710996879163 - 0.273914875291235j,
    (-2, 3, 0): 0.599443288437491 - 0.092703047944948j,
}


def schwarzschild_qnm_reference(
    mode: SeparatedMode,
    frequency_tolerance: float,
    separation_tolerance: float,
    /,
) -> QnmReferenceMode:
    """Return a versioned Schwarzschild ``M omega`` regression reference.

    The reference is not selected implicitly by :func:`solve_qnm`; supplying it
    there is the explicit qualification request.
    """

    if not isinstance(mode, SeparatedMode):
        raise TypeError("mode must be a SeparatedMode.")
    key = mode.spin_weight, mode.ell, mode.overtone
    if key not in _SCHWARZSCHILD_REFERENCE_FREQUENCIES:
        raise ValueError("No built-in Schwarzschild reference for this (s, ell, n).")
    separation = mode.ell * (mode.ell + 1) - mode.spin_weight * (mode.spin_weight + 1)
    return QnmReferenceMode(
        mode,
        jnp.asarray(0.0),
        jnp.asarray(_SCHWARZSCHILD_REFERENCE_FREQUENCIES[key]),
        jnp.asarray(complex(separation, 0.0)),
        frequency_tolerance,
        separation_tolerance,
        source_id="schwarzschild-leaver-Momega-reference",
    )


def _real_scalar(value: ArrayLike, name: str, /) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != () or not jnp.issubdtype(scalar.dtype, jnp.floating):
        raise TypeError(f"{name} must be one real floating scalar.")
    return scalar


def _complex_scalar(value: ArrayLike, name: str, /) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != () or not jnp.issubdtype(scalar.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be one complex floating scalar.")
    return scalar


def _continued_fraction_value(
    alpha: Array,
    beta: Array,
    gamma: Array,
    depth: int,
    inversion_index: int,
    /,
) -> Array:
    """Evaluate one finite inverted three-term continued fraction."""

    left_denominator = beta[0]
    for index in range(1, inversion_index):
        left_denominator = beta[index] - (
            alpha[index - 1] * gamma[index] / left_denominator
        )
    left_fraction = (
        jnp.zeros_like(beta[0])
        if inversion_index == 0
        else alpha[inversion_index - 1] * gamma[inversion_index] / left_denominator
    )

    right_denominator = beta[depth]
    for index in range(depth - 1, inversion_index, -1):
        right_denominator = beta[index] - (
            alpha[index] * gamma[index + 1] / right_denominator
        )
    right_fraction = (
        alpha[inversion_index] * gamma[inversion_index + 1] / right_denominator
    )
    return beta[inversion_index] - left_fraction - right_fraction


def _continued_fraction_evidence(
    coefficient_function,
    plan: BoundedContinuedFractionPlan,
    /,
) -> ContinuedFractionDepthEvidence:
    alpha, beta, gamma = coefficient_function(plan.maximum_depth)
    value = _continued_fraction_value(
        alpha,
        beta,
        gamma,
        plan.maximum_depth,
        plan.inversion_index,
    )
    comparison_value = _continued_fraction_value(
        alpha,
        beta,
        gamma,
        plan.comparison_depth,
        plan.inversion_index,
    )
    error = jnp.abs(value - comparison_value)
    threshold = plan.absolute_tolerance + plan.relative_tolerance * jnp.maximum(
        jnp.abs(value), 1.0
    )
    finite = (
        jnp.isfinite(jnp.real(value))
        & jnp.isfinite(jnp.imag(value))
        & jnp.isfinite(jnp.real(comparison_value))
        & jnp.isfinite(jnp.imag(comparison_value))
        & jnp.isfinite(error)
    )
    return ContinuedFractionDepthEvidence(
        value,
        comparison_value,
        error,
        threshold,
        finite,
        finite & (error <= threshold),
        jnp.asarray(plan.comparison_depth, dtype=jnp.int32),
        jnp.asarray(plan.maximum_depth, dtype=jnp.int32),
        plan.plan_id,
    )


def _angular_coefficients(
    plan: QnmSolvePlan,
    angular_frequency: Array,
    separation_constant: Array,
    dimensionless_spin: Array,
    depth: int,
    /,
) -> tuple[Array, Array, Array]:
    mode = plan.mode
    c = dimensionless_spin * angular_frequency
    dtype = jnp.result_type(c, separation_constant)
    n = jnp.arange(depth + 1, dtype=jnp.real(c).dtype).astype(dtype)
    k1 = 0.5 * abs(mode.m - mode.spin_weight)
    k2 = 0.5 * abs(mode.m + mode.spin_weight)
    total = k1 + k2
    alpha = -2.0 * (n + 1.0) * (n + 2.0 * k1 + 1.0)
    beta = (
        n * (n - 1.0)
        + 2.0 * n * (total + 1.0 - 2.0 * c)
        - 2.0 * c * (2.0 * k1 + mode.spin_weight + 1.0)
        + total * (total + 1.0)
        - c * c
        - mode.spin_weight * (mode.spin_weight + 1.0)
        - separation_constant
    )
    gamma = 2.0 * c * (n + total + mode.spin_weight)
    return alpha, beta, gamma


def _radial_coefficients(
    plan: QnmSolvePlan,
    angular_frequency: Array,
    separation_constant: Array,
    dimensionless_spin: Array,
    depth: int,
    /,
) -> tuple[Array, Array, Array]:
    mode = plan.mode
    dtype = jnp.result_type(angular_frequency, separation_constant)
    n = jnp.arange(depth + 1, dtype=jnp.real(angular_frequency).dtype).astype(dtype)
    leaver_frequency = 2.0 * angular_frequency
    leaver_spin = 0.5 * dimensionless_spin
    horizon_gap = jnp.sqrt(1.0 - dimensionless_spin * dimensionless_spin).astype(dtype)
    detuning = 0.5 * leaver_frequency - leaver_spin * mode.m
    c0 = 1.0 - mode.spin_weight - 1.0j * leaver_frequency - 2.0j * detuning / horizon_gap
    c1 = (
        -4.0
        + 2.0j * leaver_frequency * (2.0 + horizon_gap)
        + 4.0j * detuning / horizon_gap
    )
    c2 = mode.spin_weight + 3.0 - 3.0j * leaver_frequency - 2.0j * detuning / horizon_gap
    c3 = (
        leaver_frequency**2 * (4.0 + 2.0 * horizon_gap - leaver_spin**2)
        - 2.0 * leaver_spin * mode.m * leaver_frequency
        - mode.spin_weight
        - 1.0
        + (2.0 + horizon_gap) * 1.0j * leaver_frequency
        - separation_constant
        + (4.0 * leaver_frequency + 2.0j) * detuning / horizon_gap
    )
    c4 = (
        mode.spin_weight
        + 1.0
        - 2.0 * leaver_frequency**2
        - (2.0 * mode.spin_weight + 3.0) * 1.0j * leaver_frequency
        - (4.0 * leaver_frequency + 2.0j) * detuning / horizon_gap
    )
    alpha = n * n + (c0 + 1.0) * n + c0
    beta = -2.0 * n * n + (c1 + 2.0) * n + c3
    gamma = n * n + (c2 - 3.0) * n + c4 - c2 + 2.0
    return alpha, beta, gamma


def _depth_evidence(
    plan: QnmSolvePlan,
    angular_frequency: Array,
    separation_constant: Array,
    dimensionless_spin: Array,
    /,
) -> tuple[ContinuedFractionDepthEvidence, ContinuedFractionDepthEvidence]:
    angular = _continued_fraction_evidence(
        lambda depth: _angular_coefficients(
            plan,
            angular_frequency,
            separation_constant,
            dimensionless_spin,
            depth,
        ),
        plan.angular_fraction,
    )
    radial = _continued_fraction_evidence(
        lambda depth: _radial_coefficients(
            plan,
            angular_frequency,
            separation_constant,
            dimensionless_spin,
            depth,
        ),
        plan.radial_fraction,
    )
    return angular, radial


def _state_from_complex(
    angular_frequency: Array,
    separation_constant: Array,
    /,
) -> Array:
    real_dtype = jnp.result_type(
        jnp.real(angular_frequency), jnp.real(separation_constant)
    )
    return jnp.asarray(
        (
            jnp.real(angular_frequency),
            jnp.imag(angular_frequency),
            jnp.real(separation_constant),
            jnp.imag(separation_constant),
        ),
        dtype=real_dtype,
    )


def _complex_from_state(state: Array, /) -> tuple[Array, Array]:
    frequency = jax.lax.complex(state[0], state[1])
    separation = jax.lax.complex(state[2], state[3])
    return frequency, separation


def _real_residual(
    state: Array, dimensionless_spin: Array, plan: QnmSolvePlan, /
) -> Array:
    frequency, separation = _complex_from_state(state)
    angular, radial = _depth_evidence(plan, frequency, separation, dimensionless_spin)
    return jnp.asarray(
        (
            jnp.real(angular.value),
            jnp.imag(angular.value),
            jnp.real(radial.value),
            jnp.imag(radial.value),
        ),
        dtype=state.dtype,
    )


def qnm_continuation_problem(plan: QnmSolvePlan, /) -> ParameterContinuationProblem:
    """Adapt this exact coupled residual to the native spin-continuation runtime."""

    if not isinstance(plan, QnmSolvePlan):
        raise TypeError("plan must be a QnmSolvePlan.")
    dtype = plan.dimensionless_spin.dtype
    space = ArraySpace((4,), dtype=dtype, space_id=f"{plan.plan_id}:qnm-root-space")
    return ParameterContinuationProblem(
        lambda state, spin, args: _real_residual(state, spin, plan),
        parameter_lower=plan.continuation_lower,
        parameter_upper=plan.continuation_upper,
        state_space=space,
        residual_space=space,
        problem_id=f"{plan.plan_id}:spin-continuation",
    )


def _root_problem(plan: QnmSolvePlan, dtype, /) -> NonlinearSystemProblem:
    space = ArraySpace((4,), dtype=dtype, space_id=f"{plan.plan_id}:qnm-root-space")
    return NonlinearSystemProblem(
        lambda state, spin: _real_residual(state, spin, plan),
        state_space=space,
        residual_space=space,
        problem_id=f"{plan.plan_id}:coupled-root",
    )


def _root_condition(
    problem: NonlinearSystemProblem,
    state: Array,
    spin: Array,
    plan: QnmSolvePlan,
    /,
) -> tuple[Array, Array, Array]:
    jacobian = jax.jacfwd(lambda value: problem.residual(value, spin))(state)
    operator = DenseLinearOperator(
        jacobian,
        operator_id=f"{plan.plan_id}:coupled-root-jacobian",
    )
    decomposition = svd_api.svd(
        svd_api.SVDProblem(
            operator,
            problem_id=f"{plan.plan_id}:coupled-root-conditioning",
        ),
        policy=svd_api.SVDSolvePolicy(count=4, which="largest"),
    )
    values = decomposition.singular_values
    maximum = jnp.max(values)
    minimum = jnp.min(values)
    condition = maximum / jnp.maximum(minimum, jnp.finfo(values.dtype).tiny)
    finite = (
        decomposition.successful & jnp.all(jnp.isfinite(values)) & jnp.isfinite(condition)
    )
    return condition, minimum, finite


def _radial_resolution(
    plan: QnmSolvePlan,
    angular_frequency: Array,
    separation_constant: Array,
    /,
) -> SchwarzschildRadialResult | KerrTeukolskyRadialResult:
    physical_frequency = angular_frequency / plan.radial_plan.mass.astype(
        angular_frequency.dtype
    )
    if isinstance(plan.radial_plan, SchwarzschildRadialPlan):
        return evaluate_schwarzschild_radial(
            plan.radial_plan,
            physical_frequency,
            separation_constant,
        )
    return evaluate_kerr_teukolsky_radial(
        plan.radial_plan,
        physical_frequency,
        separation_constant,
    )


def solve_qnm(
    plan: QnmSolvePlan,
    initial_angular_frequency: ArrayLike,
    initial_separation_constant: ArrayLike,
    /,
    *,
    qualification: QnmReferenceMode | None = None,
    continuation_active: ArrayLike = False,
) -> QnmResult:
    """Solve one explicitly seeded coupled angular/radial QNM root."""

    if not isinstance(plan, QnmSolvePlan):
        raise TypeError("plan must be a QnmSolvePlan.")
    if qualification is not None and not isinstance(qualification, QnmReferenceMode):
        raise TypeError("qualification must be a QnmReferenceMode or None.")
    frequency_seed = _complex_scalar(
        initial_angular_frequency, "initial_angular_frequency"
    )
    separation_seed = _complex_scalar(
        initial_separation_constant, "initial_separation_constant"
    )
    active = jnp.asarray(continuation_active, dtype=jnp.bool_)
    if active.shape != ():
        raise ValueError("continuation_active must be one Boolean scalar.")
    initial_state = _state_from_complex(frequency_seed, separation_seed)
    problem = _root_problem(plan, initial_state.dtype)
    nonlinear = root(
        problem,
        initial_state,
        method=plan.nonlinear_method,
        termination=plan.termination,
        args=plan.dimensionless_spin.astype(initial_state.dtype),
    )
    frequency, separation = _complex_from_state(nonlinear.state)
    angular_depth, radial_depth = _depth_evidence(
        plan,
        frequency,
        separation,
        plan.dimensionless_spin.astype(initial_state.dtype),
    )
    residuals = jnp.asarray((angular_depth.value, radial_depth.value))
    residual_norm = jnp.max(jnp.abs(residuals))

    spheroidicity = plan.dimensionless_spin.astype(frequency.dtype) * frequency
    angular_resolution = solve_spheroidal_angular(plan.angular_plan, spheroidicity)
    angular_resolution_error = jnp.abs(
        separation - angular_resolution.separation_constant
    )
    angular_scale = jnp.maximum(jnp.abs(separation), 1.0)
    angular_resolved = angular_resolution.qualified & (
        angular_resolution_error <= plan.angular_resolution_tolerance * angular_scale
    )
    radial_resolution = _radial_resolution(plan, frequency, separation)
    radial_resolution_error = jnp.abs(radial_resolution.residual * plan.radial_plan.mass)

    condition, minimum_singular_value, condition_finite = _root_condition(
        problem,
        nonlinear.state,
        plan.dimensionless_spin.astype(initial_state.dtype),
        plan,
    )
    simple_root = condition_finite & (condition <= plan.condition_limit)
    sensitivity = root_solution_jvp(
        problem,
        nonlinear.state,
        plan.dimensionless_spin.astype(initial_state.dtype),
        jnp.ones((), dtype=initial_state.dtype),
        policy=plan.sensitivity,
    )
    derivative_state = sensitivity.value
    frequency_derivative, separation_derivative = _complex_from_state(derivative_state)

    finite = (
        jnp.isfinite(jnp.real(frequency))
        & jnp.isfinite(jnp.imag(frequency))
        & jnp.isfinite(jnp.real(separation))
        & jnp.isfinite(jnp.imag(separation))
        & jnp.all(jnp.isfinite(jnp.real(residuals)))
        & jnp.all(jnp.isfinite(jnp.imag(residuals)))
        & angular_depth.finite
        & radial_depth.finite
        & angular_resolution.finite
        & radial_resolution.finite
        & condition_finite
    )
    depth_resolved = angular_depth.resolved & radial_depth.resolved
    radial_resolved = radial_resolution.qualified
    converged = (
        nonlinear.successful
        & depth_resolved
        & angular_resolved
        & radial_resolved
        & finite
    )
    physically_valid = (
        finite
        & (jnp.imag(frequency) < 0.0)
        & (jnp.abs(plan.dimensionless_spin) < 1.0)
        & radial_resolution.physically_valid
    )

    reference_frequency_error = jnp.asarray(jnp.nan, dtype=jnp.real(frequency).dtype)
    reference_separation_error = jnp.asarray(jnp.nan, dtype=jnp.real(frequency).dtype)
    qualification_source_id = ""
    reference_id = ""
    qualified_by_reference = jnp.asarray(False)
    if qualification is not None:
        qualification_source_id = qualification.source_id
        reference_id = qualification.reference_id
        reference_frequency_error = jnp.abs(frequency - qualification.angular_frequency)
        reference_separation_error = jnp.abs(
            separation - qualification.separation_constant
        )
        spin_scale = jnp.maximum(jnp.abs(plan.dimensionless_spin), 1.0)
        spin_matches = (
            jnp.abs(plan.dimensionless_spin - qualification.dimensionless_spin)
            <= 64.0 * jnp.finfo(plan.dimensionless_spin.dtype).eps * spin_scale
        )
        qualified_by_reference = (
            jnp.asarray(qualification.mode_id == plan.mode.mode_id)
            & spin_matches
            & (reference_frequency_error <= qualification.frequency_tolerance)
            & (reference_separation_error <= qualification.separation_tolerance)
        )
    qualified = converged & physically_valid & qualified_by_reference

    derivative_finite = (
        jnp.isfinite(jnp.real(frequency_derivative))
        & jnp.isfinite(jnp.imag(frequency_derivative))
        & jnp.isfinite(jnp.real(separation_derivative))
        & jnp.isfinite(jnp.imag(separation_derivative))
    )
    derivative_valid = (
        converged
        & physically_valid
        & simple_root
        & sensitivity.evidence.successful
        & radial_resolution.derivative_valid
        & derivative_finite
    )
    derivative_status = jnp.where(
        ~finite | ~derivative_finite,
        int(QnmDerivativeStatus.NONFINITE),
        jnp.where(
            ~nonlinear.successful,
            int(QnmDerivativeStatus.PRIMAL_NOT_CONVERGED),
            jnp.where(
                ~depth_resolved,
                int(QnmDerivativeStatus.DEPTH_UNRESOLVED),
                jnp.where(
                    ~angular_resolved,
                    int(QnmDerivativeStatus.ANGULAR_BRANCH_UNRESOLVED),
                    jnp.where(
                        ~radial_resolved,
                        int(QnmDerivativeStatus.RADIAL_RESOLUTION_UNRESOLVED),
                        jnp.where(
                            ~radial_resolution.derivative_valid,
                            int(QnmDerivativeStatus.RADIAL_DERIVATIVE_UNRESOLVED),
                            jnp.where(
                                ~simple_root,
                                int(QnmDerivativeStatus.NONSIMPLE_ROOT),
                                jnp.where(
                                    ~sensitivity.evidence.successful,
                                    int(QnmDerivativeStatus.SENSITIVITY_SOLVE_FAILED),
                                    int(QnmDerivativeStatus.VALID),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)

    status = jnp.where(
        ~finite,
        int(QnmStatus.NONFINITE),
        jnp.where(
            ~nonlinear.successful,
            int(QnmStatus.NONLINEAR_NOT_CONVERGED),
            jnp.where(
                ~depth_resolved,
                int(QnmStatus.CONTINUED_FRACTION_DEPTH_UNRESOLVED),
                jnp.where(
                    ~angular_resolved,
                    int(QnmStatus.ANGULAR_RESOLUTION_UNRESOLVED),
                    jnp.where(
                        ~radial_resolved,
                        int(QnmStatus.RADIAL_RESOLUTION_UNRESOLVED),
                        jnp.where(
                            jnp.imag(frequency) >= 0.0,
                            int(QnmStatus.NONDECAYING_MODE),
                            jnp.where(
                                ~simple_root,
                                int(QnmStatus.NONSIMPLE_ROOT),
                                int(QnmStatus.SUCCESS),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)

    return QnmResult(
        frequency,
        separation,
        residuals,
        angular_depth.value,
        radial_depth.value,
        residual_norm,
        angular_depth,
        radial_depth,
        angular_resolution,
        angular_resolution_error,
        radial_resolution,
        radial_resolution_error,
        condition,
        minimum_singular_value,
        frequency_derivative,
        separation_derivative,
        sensitivity.evidence,
        nonlinear,
        reference_frequency_error,
        reference_separation_error,
        plan.dimensionless_spin,
        active,
        finite,
        converged,
        physically_valid,
        qualified,
        derivative_valid,
        status,
        derivative_status,
        plan.mode.mode_id,
        plan.radial_source_id,
        qualification_source_id,
        reference_id,
        plan.branch_id,
        plan.plan_id,
    )


__all__ = [
    "BoundedContinuedFractionPlan",
    "ContinuedFractionDepthEvidence",
    "QnmDerivativeStatus",
    "QnmReferenceMode",
    "QnmResult",
    "QnmSolvePlan",
    "QnmStatus",
    "qnm_continuation_problem",
    "schwarzschild_qnm_reference",
    "solve_qnm",
]
