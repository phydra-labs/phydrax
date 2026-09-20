#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy.optimize import brentq, linprog

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...linalg._certificates import SpectralInterval
from ...linalg._rational_functions import PartialFractionRationalFunction


RationalErrorMetric: TypeAlias = Literal["absolute", "relative"]


class RationalApproximationTarget(StrictModule):
    """Positive product target ``prod_j (x + shift_j)**exponent_j``."""

    shifts: tuple[float, ...] = eqx.field(static=True)
    exponents: tuple[float, ...] = eqx.field(static=True)
    target_id: str = eqx.field(static=True)

    def __init__(
        self,
        factors: tuple[tuple[float, float], ...],
        /,
    ):
        if not factors:
            raise ValueError(
                "A rational approximation target requires at least one factor."
            )
        shifts = tuple(float(factor[0]) for factor in factors)
        exponents = tuple(float(factor[1]) for factor in factors)
        if any(not math.isfinite(value) or value < 0.0 for value in shifts):
            raise ValueError("Target shifts must be finite and non-negative.")
        if any(not math.isfinite(value) for value in exponents):
            raise ValueError("Target exponents must be finite.")
        if all(value == 0.0 for value in exponents):
            raise ValueError("At least one target exponent must be nonzero.")
        self.shifts = shifts
        self.exponents = exponents
        self.target_id = canonical_fingerprint(
            {
                "kind": "positive-product-rational-target",
                "factors": [
                    {"shift": shift, "exponent": exponent}
                    for shift, exponent in zip(shifts, exponents, strict=True)
                ],
            }
        )

    def __call__(self, value: ArrayLike, /) -> Array:
        argument = jnp.asarray(value)
        result = jnp.ones_like(argument, dtype=jnp.result_type(argument, jnp.float32))
        for shift, exponent in zip(self.shifts, self.exponents, strict=True):
            result = result * (argument + shift) ** exponent
        return result


def power_rational_target(
    exponent: float,
    /,
    *,
    shift: float = 0.0,
) -> RationalApproximationTarget:
    """Declare ``(x + shift)**exponent`` as a canonical rational target."""
    return RationalApproximationTarget(((shift, exponent),))


class RationalApproximationResourcePolicy(StrictModule):
    """Hard host-generation bounds checked before every coefficient allocation."""

    maximum_poles: int = eqx.field(static=True)
    maximum_verification_points: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_poles: int = 128,
        maximum_verification_points: int = 131_072,
        maximum_workspace_bytes: int = 512 * 1024 * 1024,
    ):
        poles = int(maximum_poles)
        points = int(maximum_verification_points)
        workspace = int(maximum_workspace_bytes)
        if poles < 1 or points < 32 or workspace < 1:
            raise ValueError("Rational resource bounds must be positive and nontrivial.")
        self.maximum_poles = poles
        self.maximum_verification_points = points
        self.maximum_workspace_bytes = workspace
        self.policy_id = canonical_fingerprint(
            {
                "kind": "rational-approximation-resource-policy",
                "maximum_poles": poles,
                "maximum_verification_points": points,
                "maximum_workspace_bytes": workspace,
            }
        )


class RationalApproximationPlan(StrictModule):
    """Immutable fixed-size minimax coefficient-generation plan."""

    target: RationalApproximationTarget = eqx.field(static=True)
    resources: RationalApproximationResourcePolicy = eqx.field(static=True)
    num_poles: int = eqx.field(static=True)
    verification_points: int = eqx.field(static=True)
    pole_span: float = eqx.field(static=True)
    error_metric: RationalErrorMetric = eqx.field(static=True)
    requested_tolerance: float | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        target: RationalApproximationTarget,
        /,
        *,
        num_poles: int,
        verification_points: int = 16_385,
        pole_span: float = 16.0,
        error_metric: RationalErrorMetric = "relative",
        requested_tolerance: float | None = None,
        resources: RationalApproximationResourcePolicy | None = None,
    ):
        if not isinstance(target, RationalApproximationTarget):
            raise TypeError("target must be a RationalApproximationTarget.")
        poles = int(num_poles)
        points = int(verification_points)
        span = float(pole_span)
        tolerance = None if requested_tolerance is None else float(requested_tolerance)
        policy = RationalApproximationResourcePolicy() if resources is None else resources
        if not isinstance(policy, RationalApproximationResourcePolicy):
            raise TypeError(
                "resources must be a RationalApproximationResourcePolicy or None."
            )
        if poles < 1:
            raise ValueError("num_poles must be positive.")
        if points < max(129, 16 * (poles + 2)):
            raise ValueError("verification_points is too small for the requested order.")
        if not math.isfinite(span) or span <= 1.0:
            raise ValueError("pole_span must be finite and greater than one.")
        if error_metric not in ("absolute", "relative"):
            raise ValueError("error_metric must be 'absolute' or 'relative'.")
        if tolerance is not None and (not math.isfinite(tolerance) or tolerance < 0.0):
            raise ValueError("requested_tolerance must be finite and non-negative.")
        workspace = 16 * points * (poles + 3) + 8 * (2 * points) * (poles + 2)
        if poles > policy.maximum_poles:
            raise MemoryError("Rational pole count exceeds its resource policy.")
        if points > policy.maximum_verification_points:
            raise MemoryError("Rational verification grid exceeds its resource policy.")
        if workspace > policy.maximum_workspace_bytes:
            raise MemoryError(
                "Rational generation workspace exceeds its resource policy."
            )
        self.target = target
        self.resources = policy
        self.num_poles = poles
        self.verification_points = points
        self.pole_span = span
        self.error_metric = error_metric
        self.requested_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-pole-remez-rational-plan",
                "target": target.target_id,
                "num_poles": poles,
                "verification_points": points,
                "pole_span": span,
                "error_metric": error_metric,
                "requested_tolerance": tolerance,
                "resources": policy.policy_id,
            }
        )


class CertifiedRationalApproximation(StrictModule):
    """Partial fraction together with measured whole-interval error evidence."""

    target: RationalApproximationTarget = eqx.field(static=True)
    spectral_interval: SpectralInterval
    function: PartialFractionRationalFunction
    extremal_points: Array
    extremal_errors: Array
    maximum_absolute_error: Array
    maximum_relative_error: Array
    witness: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    metric: RationalErrorMetric = eqx.field(static=True)
    verification_points: int = eqx.field(static=True)
    evidence: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        target: RationalApproximationTarget,
        spectral_interval: SpectralInterval,
        function: PartialFractionRationalFunction,
        extremal_points: ArrayLike,
        extremal_errors: ArrayLike,
        maximum_absolute_error: ArrayLike,
        maximum_relative_error: ArrayLike,
        witness: ArrayLike,
        successful: ArrayLike,
        /,
        *,
        plan_id: str,
        metric: RationalErrorMetric,
        verification_points: int,
        evidence: str,
    ):
        if not isinstance(target, RationalApproximationTarget):
            raise TypeError("target must be a RationalApproximationTarget.")
        if not isinstance(spectral_interval, SpectralInterval):
            raise TypeError("spectral_interval must be a SpectralInterval.")
        if not isinstance(function, PartialFractionRationalFunction):
            raise TypeError("function must be a PartialFractionRationalFunction.")
        points = jnp.asarray(extremal_points)
        errors = jnp.asarray(extremal_errors)
        absolute = jnp.asarray(maximum_absolute_error)
        relative = jnp.asarray(maximum_relative_error)
        witness_ = jnp.asarray(witness)
        successful_ = jnp.asarray(successful, dtype=jnp.bool_)
        if points.ndim != 1 or errors.shape != points.shape or points.size < 2:
            raise ValueError(
                "Extremal points and errors must be matching nonempty vectors."
            )
        if any(
            value.shape != () for value in (absolute, relative, witness_, successful_)
        ):
            raise ValueError("Rational certificate summary values must be scalar.")
        if metric not in ("absolute", "relative"):
            raise ValueError("Unknown rational certificate metric.")
        plan_identifier = str(plan_id)
        evidence_ = str(evidence)
        if not plan_identifier or not evidence_:
            raise ValueError("Rational certificate IDs and evidence must be nonempty.")
        self.target = target
        self.spectral_interval = spectral_interval
        self.function = function
        self.extremal_points = points
        self.extremal_errors = errors
        self.maximum_absolute_error = absolute
        self.maximum_relative_error = relative
        self.witness = witness_
        self.successful = successful_
        self.plan_id = plan_identifier
        self.metric = metric
        self.verification_points = int(verification_points)
        self.evidence = evidence_
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "certified-minimax-rational-approximation",
                "target": target.target_id,
                "interval": spectral_interval.certificate_id,
                "function": function.function_id,
                "plan": plan_identifier,
                "metric": metric,
                "verification_points": int(verification_points),
                "measurements": array_tree_fingerprint(
                    (points, errors, absolute, relative, witness_, successful_)
                ),
                "evidence": evidence_,
            }
        )

    def __call__(self, value: ArrayLike, /) -> Array:
        """Evaluate the certified partial fraction pointwise."""
        return self.function(value)


def plan_minimax_rational_approximation(
    target: RationalApproximationTarget,
    /,
    *,
    num_poles: int,
    verification_points: int = 16_385,
    pole_span: float = 16.0,
    error_metric: RationalErrorMetric = "relative",
    requested_tolerance: float | None = None,
    resources: RationalApproximationResourcePolicy | None = None,
) -> RationalApproximationPlan:
    """Plan one bounded fixed-pole Remez exchange problem."""
    return RationalApproximationPlan(
        target,
        num_poles=num_poles,
        verification_points=verification_points,
        pole_span=pole_span,
        error_metric=error_metric,
        requested_tolerance=requested_tolerance,
        resources=resources,
    )


def generate_minimax_rational_approximation(
    spectral_interval: SpectralInterval,
    plan: RationalApproximationPlan,
    /,
) -> CertifiedRationalApproximation:
    """Solve the finite minimax problem and certify its measured interval maximum.

    Negative real poles are fixed geometrically from the certified interval. The
    polynomial constant and all residues are then obtained from the Chebyshev
    exchange-grid linear program, not from tabulated coefficients. Stationary
    points of the resulting analytic error are located independently and included
    in the reported maximum.
    """
    if not isinstance(spectral_interval, SpectralInterval):
        raise TypeError("spectral_interval must be a SpectralInterval.")
    if not isinstance(plan, RationalApproximationPlan):
        raise TypeError("plan must be a RationalApproximationPlan.")
    lower = float(np.asarray(jax.device_get(spectral_interval.lower)))
    upper = float(np.asarray(jax.device_get(spectral_interval.upper)))
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower <= 0.0
        or upper <= lower
    ):
        raise ValueError(
            "Rational minimax generation requires a positive nondegenerate interval."
        )

    grid = _chebyshev_grid(lower, upper, plan.verification_points)
    target = _target_numpy(plan.target, grid)
    shifts = np.geomspace(
        lower / plan.pole_span,
        upper * plan.pole_span,
        plan.num_poles,
        dtype=np.float64,
    )
    basis = np.concatenate(
        (
            np.ones((grid.size, 1), dtype=np.float64),
            1.0 / (grid[:, None] + shifts[None, :]),
        ),
        axis=1,
    )
    weights = (
        np.ones_like(target) if plan.error_metric == "absolute" else 1.0 / np.abs(target)
    )
    weighted_basis = weights[:, None] * basis
    weighted_target = weights * target
    inequalities = np.concatenate(
        (
            np.concatenate((weighted_basis, -np.ones((grid.size, 1))), axis=1),
            np.concatenate((-weighted_basis, -np.ones((grid.size, 1))), axis=1),
        ),
        axis=0,
    )
    bounds = np.concatenate((weighted_target, -weighted_target), axis=0)
    objective = np.zeros((plan.num_poles + 2,), dtype=np.float64)
    objective[-1] = 1.0
    solution = linprog(
        objective,
        A_ub=inequalities,
        b_ub=bounds,
        bounds=[(None, None)] * (plan.num_poles + 1) + [(0.0, None)],
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"Rational minimax exchange solve failed: {solution.message}")
    coefficients = np.asarray(solution.x[:-1], dtype=np.float64)
    coefficient_dtype = spectral_interval.lower.dtype
    function = PartialFractionRationalFunction(
        jnp.asarray(-shifts, dtype=coefficient_dtype),
        jnp.asarray(-coefficients[1:], dtype=coefficient_dtype),
        polynomial_coefficients=jnp.asarray((coefficients[0],), dtype=coefficient_dtype),
    )
    actual_poles = np.asarray(jax.device_get(function.poles), dtype=np.float64)
    actual_residues = np.asarray(jax.device_get(function.residues), dtype=np.float64)
    actual_constant = float(
        np.asarray(jax.device_get(function.polynomial_coefficients[0]))
    )
    actual_shifts = -actual_poles
    actual_weights = -actual_residues
    candidates = _error_stationary_candidates(
        plan.target,
        actual_constant,
        actual_shifts,
        actual_weights,
        lower,
        upper,
        grid,
        plan.error_metric,
    )
    candidate_target = _target_numpy(plan.target, candidates)
    candidate_values = actual_constant + np.sum(
        actual_weights[None, :] / (candidates[:, None] + actual_shifts[None, :]),
        axis=1,
    )
    absolute_errors = candidate_values - candidate_target
    relative_errors = absolute_errors / candidate_target
    measured = relative_errors if plan.error_metric == "relative" else absolute_errors
    extremal_indices = _extremal_indices(measured, plan.num_poles + 2)
    extremal_points = candidates[extremal_indices]
    extremal_errors = measured[extremal_indices]
    absolute_index = int(np.argmax(np.abs(absolute_errors)))
    relative_index = int(np.argmax(np.abs(relative_errors)))
    metric_index = int(np.argmax(np.abs(measured)))
    maximum_absolute_error = float(np.abs(absolute_errors[absolute_index]))
    maximum_relative_error = float(np.abs(relative_errors[relative_index]))
    measured_maximum = (
        maximum_relative_error
        if plan.error_metric == "relative"
        else maximum_absolute_error
    )
    satisfied = (
        plan.requested_tolerance is None or measured_maximum <= plan.requested_tolerance
    )
    finite = bool(
        np.all(np.isfinite(coefficients))
        and np.all(np.isfinite(measured))
        and math.isfinite(measured_maximum)
    )
    return CertifiedRationalApproximation(
        plan.target,
        spectral_interval,
        function,
        extremal_points,
        extremal_errors,
        maximum_absolute_error,
        maximum_relative_error,
        candidates[metric_index],
        finite and satisfied,
        plan_id=plan.plan_id,
        metric=plan.error_metric,
        verification_points=plan.verification_points,
        evidence="Chebyshev exchange grid plus all bracketed stationary-error points",
    )


def _chebyshev_grid(lower: float, upper: float, count: int, /) -> np.ndarray:
    angles = np.linspace(0.0, np.pi, count, dtype=np.float64)
    points = 0.5 * (lower + upper) + 0.5 * (upper - lower) * np.cos(angles)
    return np.sort(points)


def _target_numpy(
    target: RationalApproximationTarget, value: np.ndarray, /
) -> np.ndarray:
    result = np.ones_like(value, dtype=np.float64)
    for shift, exponent in zip(target.shifts, target.exponents, strict=True):
        result *= np.power(value + shift, exponent)
    return result


def _target_derivative_numpy(
    target: RationalApproximationTarget,
    value: np.ndarray | float,
    /,
) -> np.ndarray | float:
    result = _target_numpy(target, np.asarray(value, dtype=np.float64))
    logarithmic = np.zeros_like(result)
    for shift, exponent in zip(target.shifts, target.exponents, strict=True):
        logarithmic += exponent / (np.asarray(value) + shift)
    return result * logarithmic


def _error_stationary_candidates(
    target: RationalApproximationTarget,
    constant: float,
    shifts: np.ndarray,
    weights: np.ndarray,
    lower: float,
    upper: float,
    grid: np.ndarray,
    metric: RationalErrorMetric,
    /,
) -> np.ndarray:
    def derivative(value: float) -> float:
        target_value = float(_target_numpy(target, np.asarray(value)))
        target_derivative = float(_target_derivative_numpy(target, value))
        rational_value = constant + float(np.sum(weights / (value + shifts)))
        rational_derivative = -float(np.sum(weights / (value + shifts) ** 2))
        if metric == "absolute":
            return rational_derivative - target_derivative
        return (
            rational_derivative * target_value - rational_value * target_derivative
        ) / (target_value * target_value)

    derivative_values = np.asarray([derivative(float(value)) for value in grid])
    roots: list[float] = []
    for index in range(grid.size - 1):
        left_value = derivative_values[index]
        right_value = derivative_values[index + 1]
        if left_value == 0.0:
            roots.append(float(grid[index]))
        elif left_value * right_value < 0.0:
            roots.append(
                float(
                    brentq(
                        derivative,
                        float(grid[index]),
                        float(grid[index + 1]),
                        xtol=np.finfo(np.float64).eps * max(1.0, abs(grid[index])),
                        rtol=4.0 * np.finfo(np.float64).eps,
                    )
                )
            )
    merged = np.concatenate(
        (grid, np.asarray(roots, dtype=np.float64), np.asarray((lower, upper)))
    )
    return np.unique(merged)


def _extremal_indices(errors: np.ndarray, count: int, /) -> np.ndarray:
    magnitude = np.abs(errors)
    local = np.flatnonzero(
        np.concatenate(
            (
                np.asarray((True,)),
                (magnitude[1:-1] >= magnitude[:-2]) & (magnitude[1:-1] >= magnitude[2:]),
                np.asarray((True,)),
            )
        )
    )
    order = local[np.argsort(magnitude[local])[::-1]]
    selected = list(order[:count])
    if len(selected) < count:
        global_order = np.argsort(magnitude)[::-1]
        for index in global_order:
            candidate = int(index)
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) == count:
                break
    return np.asarray(sorted(selected), dtype=np.int64)


__all__ = [
    "CertifiedRationalApproximation",
    "RationalApproximationPlan",
    "RationalApproximationResourcePolicy",
    "RationalApproximationTarget",
    "RationalErrorMetric",
    "generate_minimax_rational_approximation",
    "plan_minimax_rational_approximation",
    "power_rational_target",
]
