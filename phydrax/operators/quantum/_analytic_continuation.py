#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Evidence-carrying analytic continuation for finite thermal data.

Padé continuation is a rational interpolation model.  Maximum entropy and sparse
continuation are separate regularized inverse problems on a native quadrature
measure.  All three expose structure-only plans, numerical preparation, and pure
array evaluation/solve stages; none silently substitutes another method.
"""

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...integration import GaussLegendreRule
from ...linalg import (
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    RankPolicy,
)
from ._thermal_green import (
    FermionicSpectralFunction,
    FermionicSpectralPhysicality,
    GreenFunctionStatus,
    MatsubaraGreenFunction,
)


class ContinuationStatus(IntEnum):
    """Terminal status shared by analytic-continuation algorithms."""

    SUCCESS = 0
    MAXIMUM_ITERATIONS_REACHED = 1
    NONFINITE_INPUT = 2
    NONFINITE_OUTPUT = 3
    RANK_DEFICIENT = 4
    RESIDUAL_TOO_LARGE = 5
    POSITIVITY_VIOLATION = 6
    SUM_RULE_VIOLATION = 7
    CAUSALITY_VIOLATION = 8


class ContinuationEvidence(StrictModule):
    """Fit, regularization, uncertainty, and physicality evidence."""

    residual_norm: Array
    relative_residual: Array
    chi_square: Array
    regularization_value: Array
    objective_value: Array
    condition_estimate: Array
    numerical_rank: Array
    iteration_count: Array
    gradient_norm: Array
    minimum_spectral_density: Array
    spectral_sum: Array
    sum_rule_residual: Array
    uncertainty_scale: Array
    finite: Array
    converged: Array
    valid: Array
    status: Array
    method: str = eqx.field(static=True)


class SpectralGridPlan(StrictModule):
    """Structure-only bounded quadrature plan for a real-frequency interval."""

    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)
    point_count: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    persistent_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedSpectralGrid(StrictModule):
    """Native Gauss--Legendre spectral integration measure."""

    plan: SpectralGridPlan
    frequencies: Array
    weights: Array
    grid_id: str = eqx.field(static=True)


class SpectralContinuationResult(StrictModule):
    """Spectral density, pointwise uncertainty, and inference evidence."""

    grid: PreparedSpectralGrid
    density: Array
    uncertainty: Array
    evidence: ContinuationEvidence
    continuation_id: str = eqx.field(static=True)

    @property
    def valid(self) -> Array:
        return self.evidence.valid


class PadeContinuationPlan(StrictModule):
    """Immutable rational degree, rank, and resource contract."""

    maximum_samples: int = eqx.field(static=True)
    numerator_degree: int = eqx.field(static=True)
    denominator_degree: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    causality_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedPadeContinuation(StrictModule):
    """Prepared scaled rational coefficients and interpolation evidence."""

    plan: PadeContinuationPlan
    numerator: Array
    denominator: Array
    center: Array
    scale: Array
    evidence: ContinuationEvidence
    sample_representation_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class PadeEvaluation(StrictModule):
    """Retarded values and spectral inference on caller frequencies."""

    frequencies: Array
    values: Array
    spectral_density: Array
    uncertainty: Array
    evidence: ContinuationEvidence
    prepared: PreparedPadeContinuation


class MaximumEntropyPlan(StrictModule):
    """Nonnegative entropy-regularized inverse-problem plan."""

    grid: PreparedSpectralGrid
    alpha: float = eqx.field(static=True)
    sum_rule: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    gradient_tolerance: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    maximum_samples: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedMaximumEntropy(StrictModule):
    """Whitened kernel, observations, and strictly positive prior mass."""

    plan: MaximumEntropyPlan
    samples: MatsubaraGreenFunction
    design: Array
    target: Array
    prior_mass: Array
    noise: Array
    singular_values: Array
    prepared_id: str = eqx.field(static=True)


class SparseContinuationPlan(StrictModule):
    """Elastic-net sparse spectral inverse-problem plan."""

    grid: PreparedSpectralGrid
    l1_regularization: float = eqx.field(static=True)
    l2_regularization: float = eqx.field(static=True)
    nonnegative: bool = eqx.field(static=True)
    sum_rule: float | None = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    gradient_tolerance: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    maximum_samples: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedSparseContinuation(StrictModule):
    """Whitened sparse inverse problem and rank evidence."""

    plan: SparseContinuationPlan
    samples: MatsubaraGreenFunction
    design: Array
    target: Array
    noise: Array
    singular_values: Array
    prepared_id: str = eqx.field(static=True)


def _positive(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _nonnegative(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _positive_int(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _native_factor(matrix: Array, tolerance: float, /):
    return factorize(
        DenseLinearOperator(matrix),
        FactorizationPolicy("svd", rank=RankPolicy(relative_cutoff=float(tolerance))),
    )


def plan_spectral_grid(
    lower: float,
    upper: float,
    point_count: int,
    /,
    *,
    maximum_bytes: int = 64 * 1024**2,
) -> SpectralGridPlan:
    """Plan a fixed positive quadrature measure without materializing it."""

    lower_ = float(lower)
    upper_ = float(upper)
    count = _positive_int(point_count, "point_count")
    budget = _positive_int(maximum_bytes, "maximum_bytes")
    if not isfinite(lower_) or not isfinite(upper_) or upper_ <= lower_:
        raise ValueError("Spectral bounds must be finite and strictly increasing.")
    persistent = 2 * count * np.dtype(np.float64).itemsize
    if persistent > budget:
        raise ValueError("Spectral grid exceeds maximum_bytes before allocation.")
    plan_id = canonical_fingerprint(
        {
            "kind": "spectral-gauss-legendre-grid-plan",
            "lower": lower_,
            "upper": upper_,
            "point_count": count,
            "maximum_bytes": budget,
            "persistent_bytes": persistent,
        }
    )
    return SpectralGridPlan(lower_, upper_, count, budget, persistent, plan_id)


def prepare_spectral_grid(plan: SpectralGridPlan, /) -> PreparedSpectralGrid:
    """Materialize the native Gauss--Legendre integration measure."""

    if not isinstance(plan, SpectralGridPlan):
        raise TypeError("plan must be a SpectralGridPlan.")
    data = GaussLegendreRule(plan.point_count).data()
    half_width = 0.5 * (plan.upper - plan.lower)
    midpoint = 0.5 * (plan.upper + plan.lower)
    frequencies = midpoint + half_width * jnp.asarray(data.nodes)
    weights = half_width * jnp.asarray(data.weights)
    grid_id = canonical_fingerprint(
        {
            "kind": "prepared-spectral-grid",
            "plan": plan.plan_id,
            "measure": array_tree_fingerprint(
                {"frequencies": frequencies, "weights": weights}
            ),
        }
    )
    return PreparedSpectralGrid(plan, frequencies, weights, grid_id)


def evaluate_spectral_continuation(
    result: SpectralContinuationResult,
    frequency: ArrayLike,
    /,
) -> Array:
    """Evaluate the Cauchy transform of an inferred quadrature density."""

    if not isinstance(result, SpectralContinuationResult):
        raise TypeError("result must be a SpectralContinuationResult.")
    z = jnp.asarray(frequency)
    kernel = jnp.reciprocal(z[..., None] - result.grid.frequencies)
    mass = result.grid.weights * result.density
    return contract("...r,r->...", kernel, mass)


def spectral_grid(
    lower: float,
    upper: float,
    point_count: int,
    /,
    *,
    maximum_bytes: int = 64 * 1024**2,
) -> PreparedSpectralGrid:
    """Plan and prepare a bounded native spectral grid."""

    return prepare_spectral_grid(
        plan_spectral_grid(lower, upper, point_count, maximum_bytes=maximum_bytes)
    )


def plan_pade_continuation(
    maximum_samples: int,
    /,
    *,
    numerator_degree: int | None = None,
    denominator_degree: int | None = None,
    maximum_bytes: int = 64 * 1024**2,
    rank_tolerance: float = 1e-12,
    residual_tolerance: float = 1e-7,
    causality_tolerance: float = 1e-8,
) -> PadeContinuationPlan:
    """Plan a bounded multipoint Padé fit."""

    samples = _positive_int(maximum_samples, "maximum_samples")
    if numerator_degree is None and denominator_degree is None:
        denominator = max((samples - 1) // 2, 0)
        numerator = samples - denominator - 1
    elif numerator_degree is not None and denominator_degree is not None:
        numerator = int(numerator_degree)
        denominator = int(denominator_degree)
    else:
        raise ValueError(
            "numerator_degree and denominator_degree must be supplied together."
        )
    if numerator < 0 or denominator < 0:
        raise ValueError("Padé degrees must be non-negative.")
    if numerator + denominator + 1 > samples:
        raise ValueError("Padé coefficient count must not exceed maximum_samples.")
    budget = _positive_int(maximum_bytes, "maximum_bytes")
    rank_ = _positive(rank_tolerance, "rank_tolerance")
    residual_ = _positive(residual_tolerance, "residual_tolerance")
    causality_ = _nonnegative(causality_tolerance, "causality_tolerance")
    unknowns = numerator + denominator + 1
    required = np.dtype(np.complex128).itemsize * (
        samples * unknowns + 4 * unknowns * unknowns + 4 * unknowns
    )
    if required > budget:
        raise ValueError("Padé continuation exceeds maximum_bytes before allocation.")
    plan_id = canonical_fingerprint(
        {
            "kind": "multipoint-pade-plan",
            "maximum_samples": samples,
            "numerator_degree": numerator,
            "denominator_degree": denominator,
            "maximum_bytes": budget,
            "rank_tolerance": rank_,
            "residual_tolerance": residual_,
            "causality_tolerance": causality_,
        }
    )
    return PadeContinuationPlan(
        samples,
        numerator,
        denominator,
        budget,
        rank_,
        residual_,
        causality_,
        plan_id,
    )


def _active_scalar_samples(
    samples: MatsubaraGreenFunction, /
) -> tuple[Array, Array, Array]:
    if not isinstance(samples, MatsubaraGreenFunction):
        raise TypeError("samples must be a MatsubaraGreenFunction.")
    if samples.values.ndim != 1:
        raise ValueError(
            "Analytic continuation currently requires scalar Green-function data."
        )
    active = samples.sample_active
    frequency = samples.frequencies
    return frequency, samples.values, active


def _empty_evidence(
    method: str,
    finite: ArrayLike,
    /,
    *,
    residual: ArrayLike,
    relative: ArrayLike,
    condition: ArrayLike,
    rank: ArrayLike,
    converged: ArrayLike,
    status: ArrayLike,
) -> ContinuationEvidence:
    dtype = jnp.asarray(relative).real.dtype
    nan = jnp.asarray(jnp.nan, dtype=dtype)
    zero = jnp.asarray(0.0, dtype=dtype)
    valid = jnp.asarray(finite) & jnp.asarray(converged)
    return ContinuationEvidence(
        jnp.asarray(residual),
        jnp.asarray(relative),
        jnp.asarray(residual) ** 2,
        zero,
        jnp.asarray(residual) ** 2,
        jnp.asarray(condition),
        jnp.asarray(rank, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        zero,
        nan,
        nan,
        nan,
        jnp.asarray(residual),
        jnp.asarray(finite),
        jnp.asarray(converged),
        valid,
        jnp.asarray(status, dtype=jnp.int32),
        method,
    )


def prepare_pade_continuation(
    plan: PadeContinuationPlan,
    samples: MatsubaraGreenFunction,
    /,
) -> PreparedPadeContinuation:
    """Fit scaled Padé coefficients through Phydrax native dense SVD."""

    if not isinstance(plan, PadeContinuationPlan):
        raise TypeError("plan must be a PadeContinuationPlan.")
    frequency, values, active = _active_scalar_samples(samples)
    count = values.shape[0]
    unknowns = plan.numerator_degree + plan.denominator_degree + 1
    if count > plan.maximum_samples:
        raise ValueError("Padé samples exceed plan.maximum_samples.")
    if count < unknowns:
        raise ValueError("Active Padé system has fewer samples than coefficients.")
    z = 1j * frequency.astype(jnp.result_type(values, 1j))
    active_weight = active.astype(z.real.dtype)
    active_count = jnp.maximum(jnp.sum(active_weight), 1.0)
    center = jnp.sum(active_weight * z) / active_count
    scale = jnp.maximum(jnp.max(jnp.where(active, jnp.abs(z - center), 0.0)), 1.0)
    x = (z - center) / scale
    numerator_powers = x[:, None] ** jnp.arange(plan.numerator_degree + 1)
    if plan.denominator_degree:
        denominator_powers = x[:, None] ** jnp.arange(1, plan.denominator_degree + 1)
        design = jnp.concatenate(
            (numerator_powers, -values[:, None] * denominator_powers), axis=1
        )
    else:
        design = numerator_powers
    design = design * active_weight[:, None]
    target = values * active_weight
    finite_input = (
        jnp.all(jnp.isfinite(z) | ~active)
        & jnp.all(jnp.isfinite(values) | ~active)
        & samples.valid
    )
    factor = _native_factor(design, plan.rank_tolerance)
    coefficients = factor.solve(target).value
    numerator = coefficients[: plan.numerator_degree + 1]
    denominator = jnp.concatenate(
        (
            jnp.ones((1,), dtype=coefficients.dtype),
            coefficients[plan.numerator_degree + 1 :],
        )
    )
    fitted = design @ coefficients
    residual_vector = (fitted - target) * active_weight
    residual = jnp.sqrt(jnp.sum(jnp.abs(residual_vector) ** 2))
    norm = jnp.sqrt(jnp.sum(jnp.abs(target) ** 2))
    relative = residual / jnp.maximum(norm, jnp.finfo(residual.real.dtype).tiny)
    singular_values = factor.singular_values()
    rank = factor.rank()
    condition = singular_values[0] / jnp.maximum(
        singular_values[-1], jnp.finfo(singular_values.dtype).tiny
    )
    finite = finite_input & jnp.all(jnp.isfinite(coefficients)) & jnp.isfinite(relative)
    full_rank = rank == unknowns
    converged = finite & full_rank & (relative <= plan.residual_tolerance)
    status = jnp.where(
        ~finite_input,
        int(ContinuationStatus.NONFINITE_INPUT),
        jnp.where(
            ~finite,
            int(ContinuationStatus.NONFINITE_OUTPUT),
            jnp.where(
                ~full_rank,
                int(ContinuationStatus.RANK_DEFICIENT),
                jnp.where(
                    converged,
                    int(ContinuationStatus.SUCCESS),
                    int(ContinuationStatus.RESIDUAL_TOO_LARGE),
                ),
            ),
        ),
    ).astype(jnp.int32)
    evidence = _empty_evidence(
        "pade",
        finite,
        residual=residual,
        relative=relative,
        condition=condition,
        rank=rank,
        converged=converged,
        status=status,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-multipoint-pade",
            "plan": plan.plan_id,
            "samples": samples.representation_id,
            "coefficients": array_tree_fingerprint(
                {"numerator": numerator, "denominator": denominator}
            ),
        }
    )
    return PreparedPadeContinuation(
        plan,
        numerator,
        denominator,
        center,
        scale,
        evidence,
        samples.representation_id,
        prepared_id,
    )


def evaluate_pade(
    prepared: PreparedPadeContinuation,
    frequencies: ArrayLike,
    /,
    *,
    broadening: float = 1e-3,
) -> PadeEvaluation:
    """Evaluate a prepared Padé approximant on the retarded real-frequency line."""

    if not isinstance(prepared, PreparedPadeContinuation):
        raise TypeError("prepared must be a PreparedPadeContinuation.")
    eta = _positive(broadening, "broadening")
    omega = jnp.asarray(frequencies)
    z = omega + 1j * eta
    x = (z - prepared.center) / prepared.scale
    numerator_powers = x[..., None] ** jnp.arange(prepared.numerator.shape[0])
    denominator_powers = x[..., None] ** jnp.arange(prepared.denominator.shape[0])
    numerator = contract("...k,k->...", numerator_powers, prepared.numerator)
    denominator = contract("...k,k->...", denominator_powers, prepared.denominator)
    values = numerator / denominator
    density = -jnp.imag(values) / jnp.pi
    sensitivity = jnp.sqrt(
        jnp.sum(jnp.abs(numerator_powers) ** 2, axis=-1)
    ) / jnp.maximum(jnp.abs(denominator), jnp.finfo(density.dtype).tiny)
    uncertainty = prepared.evidence.relative_residual * sensitivity
    minimum = jnp.min(density)
    finite = prepared.evidence.finite & jnp.all(jnp.isfinite(values))
    causal = minimum >= -prepared.plan.causality_tolerance
    valid = prepared.evidence.valid & finite & causal
    status = jnp.where(
        ~finite,
        int(ContinuationStatus.NONFINITE_OUTPUT),
        jnp.where(
            ~causal,
            int(ContinuationStatus.CAUSALITY_VIOLATION),
            prepared.evidence.status,
        ),
    ).astype(jnp.int32)
    evidence = ContinuationEvidence(
        prepared.evidence.residual_norm,
        prepared.evidence.relative_residual,
        prepared.evidence.chi_square,
        prepared.evidence.regularization_value,
        prepared.evidence.objective_value,
        prepared.evidence.condition_estimate,
        prepared.evidence.numerical_rank,
        prepared.evidence.iteration_count,
        prepared.evidence.gradient_norm,
        minimum,
        jnp.asarray(jnp.nan, dtype=density.dtype),
        jnp.asarray(jnp.nan, dtype=density.dtype),
        jnp.max(uncertainty),
        finite,
        prepared.evidence.converged,
        valid,
        status,
        "pade",
    )
    return PadeEvaluation(omega, values, density, uncertainty, evidence, prepared)


def pade_continuation(
    samples: MatsubaraGreenFunction,
    frequencies: ArrayLike,
    /,
    *,
    numerator_degree: int | None = None,
    denominator_degree: int | None = None,
    broadening: float = 1e-3,
    maximum_bytes: int = 64 * 1024**2,
    rank_tolerance: float = 1e-12,
    residual_tolerance: float = 1e-7,
    causality_tolerance: float = 1e-8,
) -> PadeEvaluation:
    """Plan, fit, and evaluate multipoint Padé continuation."""

    plan = plan_pade_continuation(
        samples.values.shape[0],
        numerator_degree=numerator_degree,
        denominator_degree=denominator_degree,
        maximum_bytes=maximum_bytes,
        rank_tolerance=rank_tolerance,
        residual_tolerance=residual_tolerance,
        causality_tolerance=causality_tolerance,
    )
    return evaluate_pade(
        prepare_pade_continuation(plan, samples),
        frequencies,
        broadening=broadening,
    )


def _validate_grid(grid: PreparedSpectralGrid, /) -> None:
    if not isinstance(grid, PreparedSpectralGrid):
        raise TypeError("grid must be a PreparedSpectralGrid.")


def _inverse_problem_bytes(samples: int, points: int, /) -> int:
    return np.dtype(np.float64).itemsize * (
        4 * samples * points + 4 * points * points + 12 * points + 4 * samples
    )


def plan_maximum_entropy(
    grid: PreparedSpectralGrid,
    /,
    *,
    alpha: float,
    sum_rule: float = 1.0,
    maximum_iterations: int = 2000,
    gradient_tolerance: float = 1e-7,
    residual_tolerance: float = 1.0,
    learning_rate: float = 0.8,
    maximum_samples: int = 4096,
    maximum_bytes: int = 512 * 1024**2,
) -> MaximumEntropyPlan:
    """Plan a fixed-work nonnegative maximum-entropy inference."""

    _validate_grid(grid)
    alpha_ = _positive(alpha, "alpha")
    sum_ = _positive(sum_rule, "sum_rule")
    iterations = _positive_int(maximum_iterations, "maximum_iterations")
    gradient_ = _positive(gradient_tolerance, "gradient_tolerance")
    residual_ = _positive(residual_tolerance, "residual_tolerance")
    learning_ = _positive(learning_rate, "learning_rate")
    samples = _positive_int(maximum_samples, "maximum_samples")
    budget = _positive_int(maximum_bytes, "maximum_bytes")
    required = _inverse_problem_bytes(samples, grid.plan.point_count)
    if required > budget:
        raise ValueError("Maximum-entropy plan exceeds maximum_bytes before allocation.")
    plan_id = canonical_fingerprint(
        {
            "kind": "nonnegative-maximum-entropy-plan",
            "grid": grid.grid_id,
            "alpha": alpha_,
            "sum_rule": sum_,
            "maximum_iterations": iterations,
            "gradient_tolerance": gradient_,
            "residual_tolerance": residual_,
            "learning_rate": learning_,
            "maximum_samples": samples,
            "maximum_bytes": budget,
        }
    )
    return MaximumEntropyPlan(
        grid,
        alpha_,
        sum_,
        iterations,
        gradient_,
        residual_,
        learning_,
        samples,
        budget,
        plan_id,
    )


def _noise_array(noise: ArrayLike, count: int, /) -> Array:
    value = jnp.asarray(noise)
    if value.shape == ():
        value = jnp.full((count,), value)
    if value.shape != (count,):
        raise ValueError("noise must be scalar or have one value per sample.")
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        raise TypeError("noise must be real-valued.")
    return value


def _whitened_spectral_system(
    samples: MatsubaraGreenFunction,
    grid: PreparedSpectralGrid,
    noise: ArrayLike,
    /,
) -> tuple[Array, Array, Array]:
    frequency, values, active = _active_scalar_samples(samples)
    count = values.shape[0]
    noise_ = _noise_array(noise, count)
    safe_noise = jnp.where(active, noise_, 1.0)
    z = 1j * frequency[:, None]
    kernel = jnp.reciprocal(z - grid.frequencies[None, :])
    weighted = kernel / safe_noise[:, None]
    target = values / safe_noise
    mask = active.astype(weighted.real.dtype)
    design = jnp.concatenate(
        (weighted.real * mask[:, None], weighted.imag * mask[:, None])
    )
    target_real = jnp.concatenate((target.real * mask, target.imag * mask))
    return design, target_real, noise_


def prepare_maximum_entropy(
    plan: MaximumEntropyPlan,
    samples: MatsubaraGreenFunction,
    /,
    *,
    noise: ArrayLike = 1.0,
    prior: ArrayLike | None = None,
) -> PreparedMaximumEntropy:
    """Prepare a whitened spectral kernel and normalized positive prior."""

    if not isinstance(plan, MaximumEntropyPlan):
        raise TypeError("plan must be a MaximumEntropyPlan.")
    _, values, _ = _active_scalar_samples(samples)
    if values.shape[0] > plan.maximum_samples:
        raise ValueError("Maximum-entropy samples exceed plan.maximum_samples.")
    design, target, noise_ = _whitened_spectral_system(samples, plan.grid, noise)
    count = plan.grid.plan.point_count
    if prior is None:
        prior_density = jnp.ones((count,), dtype=design.dtype)
    else:
        prior_density = jnp.asarray(prior, dtype=design.dtype)
        if prior_density.shape != (count,):
            raise ValueError("prior must have one value per spectral grid point.")
    prior_host = np.asarray(prior_density)
    if not np.all(np.isfinite(prior_host)) or np.any(prior_host <= 0.0):
        raise ValueError("prior must be finite and strictly positive.")
    prior_mass = prior_density * plan.grid.weights
    prior_mass = jnp.maximum(prior_mass, jnp.finfo(prior_mass.dtype).tiny)
    prior_mass = plan.sum_rule * prior_mass / jnp.sum(prior_mass)
    factor = _native_factor(design, max(np.finfo(np.dtype(design.dtype)).eps * 64, 1e-14))
    singular_values = factor.singular_values()
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-maximum-entropy",
            "plan": plan.plan_id,
            "samples": samples.representation_id,
            "prior": array_tree_fingerprint(prior_mass),
            "noise": array_tree_fingerprint(noise_),
        }
    )
    return PreparedMaximumEntropy(
        plan,
        samples,
        design,
        target,
        prior_mass,
        noise_,
        singular_values,
        prepared_id,
    )


def _spectral_result_id(
    method: str,
    prepared_id: str,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "spectral-continuation-result",
            "method": method,
            "prepared": prepared_id,
        }
    )


def solve_maximum_entropy(
    prepared: PreparedMaximumEntropy,
    /,
) -> SpectralContinuationResult:
    """Run fixed-work exponentiated-gradient maximum entropy with a hard sum rule."""

    if not isinstance(prepared, PreparedMaximumEntropy):
        raise TypeError("prepared must be a PreparedMaximumEntropy.")
    plan = prepared.plan
    design = prepared.design
    target = prepared.target
    prior = prepared.prior_mass
    largest = jnp.maximum(prepared.singular_values[0], jnp.finfo(design.dtype).tiny)
    step = plan.learning_rate / (largest**2 * plan.sum_rule + plan.alpha + 1.0)
    tiny = jnp.finfo(design.dtype).tiny

    def iteration(_: int, mass: Array) -> Array:
        residual = design @ mass - target
        gradient = design.T @ residual + plan.alpha * jnp.log(
            jnp.maximum(mass, tiny) / prior
        )
        update = jnp.clip(-step * gradient, -20.0, 20.0)
        candidate = mass * jnp.exp(update)
        candidate = jnp.maximum(candidate, tiny)
        return plan.sum_rule * candidate / jnp.sum(candidate)

    mass = jax.lax.fori_loop(0, plan.maximum_iterations, iteration, prior)
    density = mass / plan.grid.weights
    residual_vector = design @ mass - target
    residual = jnp.sqrt(jnp.sum(residual_vector**2))
    target_norm = jnp.sqrt(jnp.sum(target**2))
    relative = residual / jnp.maximum(target_norm, jnp.finfo(target.dtype).tiny)
    log_ratio = jnp.log(jnp.maximum(mass, tiny) / prior)
    regularization = plan.alpha * jnp.sum(mass * log_ratio - mass + prior)
    chi_square = jnp.sum(residual_vector**2)
    objective = 0.5 * chi_square + regularization
    gradient = design.T @ residual_vector + plan.alpha * log_ratio
    projected_gradient = mass * (gradient - jnp.sum(mass * gradient) / plan.sum_rule)
    gradient_scale = jnp.maximum(
        1.0,
        jnp.max(jnp.abs(mass * (design.T @ target))),
    )
    gradient_norm = jnp.max(jnp.abs(projected_gradient)) / gradient_scale
    spectral_sum = contract("r,r->", plan.grid.weights, density)
    sum_residual = jnp.abs(spectral_sum - plan.sum_rule)
    hessian = design.T @ design + jnp.diag(plan.alpha / jnp.maximum(mass, tiny))
    hessian_factor = _native_factor(hessian, 64 * np.finfo(np.dtype(hessian.dtype)).eps)
    covariance = hessian_factor.solve(
        jnp.eye(hessian.shape[0], dtype=hessian.dtype)
    ).value
    uncertainty = jnp.sqrt(jnp.maximum(jnp.diag(covariance), 0.0)) / plan.grid.weights
    singular_values = hessian_factor.singular_values()
    condition = singular_values[0] / jnp.maximum(
        singular_values[-1], jnp.finfo(singular_values.dtype).tiny
    )
    rank = hessian_factor.rank()
    finite_input = (
        prepared.samples.valid
        & jnp.all(jnp.isfinite(prepared.target))
        & jnp.all(jnp.isfinite(prepared.noise))
        & jnp.all(prepared.noise > 0.0)
    )
    finite = (
        finite_input
        & jnp.all(jnp.isfinite(density))
        & jnp.all(jnp.isfinite(uncertainty))
        & jnp.isfinite(objective)
    )
    positive = jnp.min(density) >= 0.0
    sum_valid = sum_residual <= 64.0 * jnp.finfo(density.dtype).eps * plan.sum_rule
    converged = gradient_norm <= plan.gradient_tolerance
    residual_valid = relative <= plan.residual_tolerance
    valid = finite & positive & sum_valid & residual_valid & converged
    status = jnp.where(
        ~finite_input,
        int(ContinuationStatus.NONFINITE_INPUT),
        jnp.where(
            ~finite,
            int(ContinuationStatus.NONFINITE_OUTPUT),
            jnp.where(
                ~positive,
                int(ContinuationStatus.POSITIVITY_VIOLATION),
                jnp.where(
                    ~sum_valid,
                    int(ContinuationStatus.SUM_RULE_VIOLATION),
                    jnp.where(
                        ~residual_valid,
                        int(ContinuationStatus.RESIDUAL_TOO_LARGE),
                        jnp.where(
                            converged,
                            int(ContinuationStatus.SUCCESS),
                            int(ContinuationStatus.MAXIMUM_ITERATIONS_REACHED),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    evidence = ContinuationEvidence(
        residual,
        relative,
        chi_square,
        regularization,
        objective,
        condition,
        rank,
        jnp.asarray(plan.maximum_iterations, dtype=jnp.int32),
        gradient_norm,
        jnp.min(density),
        spectral_sum,
        sum_residual,
        jnp.max(uncertainty),
        finite,
        converged,
        valid,
        status,
        "maximum-entropy",
    )
    return SpectralContinuationResult(
        plan.grid,
        density,
        uncertainty,
        evidence,
        _spectral_result_id("maximum-entropy", prepared.prepared_id),
    )


def maximum_entropy_continuation(
    samples: MatsubaraGreenFunction,
    grid: PreparedSpectralGrid,
    /,
    *,
    alpha: float,
    sum_rule: float = 1.0,
    noise: ArrayLike = 1.0,
    prior: ArrayLike | None = None,
    maximum_iterations: int = 2000,
    gradient_tolerance: float = 1e-7,
    residual_tolerance: float = 1.0,
    learning_rate: float = 0.8,
    maximum_bytes: int = 512 * 1024**2,
) -> SpectralContinuationResult:
    """Plan, prepare, and solve nonnegative maximum-entropy continuation."""

    plan = plan_maximum_entropy(
        grid,
        alpha=alpha,
        sum_rule=sum_rule,
        maximum_iterations=maximum_iterations,
        gradient_tolerance=gradient_tolerance,
        residual_tolerance=residual_tolerance,
        learning_rate=learning_rate,
        maximum_samples=samples.values.shape[0],
        maximum_bytes=maximum_bytes,
    )
    prepared = prepare_maximum_entropy(plan, samples, noise=noise, prior=prior)
    return solve_maximum_entropy(prepared)


def plan_sparse_continuation(
    grid: PreparedSpectralGrid,
    /,
    *,
    l1_regularization: float,
    l2_regularization: float = 0.0,
    nonnegative: bool = False,
    sum_rule: float | None = None,
    maximum_iterations: int = 2000,
    gradient_tolerance: float = 1e-7,
    residual_tolerance: float = 1.0,
    maximum_samples: int = 4096,
    maximum_bytes: int = 512 * 1024**2,
) -> SparseContinuationPlan:
    """Plan a fixed-work elastic-net spectral inference."""

    _validate_grid(grid)
    l1 = _nonnegative(l1_regularization, "l1_regularization")
    l2 = _nonnegative(l2_regularization, "l2_regularization")
    if l1 == 0.0 and l2 == 0.0:
        raise ValueError("Sparse continuation requires positive L1 or L2 regularization.")
    sum_ = None if sum_rule is None else _positive(sum_rule, "sum_rule")
    iterations = _positive_int(maximum_iterations, "maximum_iterations")
    gradient_ = _positive(gradient_tolerance, "gradient_tolerance")
    residual_ = _positive(residual_tolerance, "residual_tolerance")
    samples = _positive_int(maximum_samples, "maximum_samples")
    budget = _positive_int(maximum_bytes, "maximum_bytes")
    required = _inverse_problem_bytes(samples, grid.plan.point_count)
    if required > budget:
        raise ValueError("Sparse continuation exceeds maximum_bytes before allocation.")
    plan_id = canonical_fingerprint(
        {
            "kind": "regularized-sparse-continuation-plan",
            "grid": grid.grid_id,
            "l1_regularization": l1,
            "l2_regularization": l2,
            "nonnegative": bool(nonnegative),
            "sum_rule": sum_,
            "maximum_iterations": iterations,
            "gradient_tolerance": gradient_,
            "residual_tolerance": residual_,
            "maximum_samples": samples,
            "maximum_bytes": budget,
        }
    )
    return SparseContinuationPlan(
        grid,
        l1,
        l2,
        bool(nonnegative),
        sum_,
        iterations,
        gradient_,
        residual_,
        samples,
        budget,
        plan_id,
    )


def prepare_sparse_continuation(
    plan: SparseContinuationPlan,
    samples: MatsubaraGreenFunction,
    /,
    *,
    noise: ArrayLike = 1.0,
) -> PreparedSparseContinuation:
    """Prepare one whitened elastic-net spectral inverse problem."""

    if not isinstance(plan, SparseContinuationPlan):
        raise TypeError("plan must be a SparseContinuationPlan.")
    _, values, _ = _active_scalar_samples(samples)
    if values.shape[0] > plan.maximum_samples:
        raise ValueError("Sparse-continuation samples exceed plan.maximum_samples.")
    design, target, noise_ = _whitened_spectral_system(samples, plan.grid, noise)
    factor = _native_factor(design, 64 * np.finfo(np.dtype(design.dtype)).eps)
    singular_values = factor.singular_values()
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-sparse-continuation",
            "plan": plan.plan_id,
            "samples": samples.representation_id,
            "noise": array_tree_fingerprint(noise_),
        }
    )
    return PreparedSparseContinuation(
        plan,
        samples,
        design,
        target,
        noise_,
        singular_values,
        prepared_id,
    )


def _soft_threshold(value: Array, threshold: Array, /) -> Array:
    return jnp.sign(value) * jnp.maximum(jnp.abs(value) - threshold, 0.0)


def _project_simplex(value: Array, total: float, /) -> Array:
    """Euclidean projection onto a nonnegative fixed-mass simplex."""

    ordered = jnp.sort(value)[::-1]
    cumulative = jnp.cumsum(ordered) - total
    divisor = jnp.arange(1, value.shape[0] + 1, dtype=value.dtype)
    support = ordered - cumulative / divisor > 0.0
    last = jnp.maximum(jnp.sum(support) - 1, 0)
    threshold = cumulative[last] / divisor[last]
    return jnp.maximum(value - threshold, 0.0)


def solve_sparse_continuation(
    prepared: PreparedSparseContinuation,
    /,
) -> SpectralContinuationResult:
    """Run fixed-work FISTA and report active-set uncertainty."""

    if not isinstance(prepared, PreparedSparseContinuation):
        raise TypeError("prepared must be a PreparedSparseContinuation.")
    plan = prepared.plan
    design = prepared.design
    target = prepared.target
    points = plan.grid.plan.point_count
    largest = jnp.maximum(prepared.singular_values[0], jnp.finfo(design.dtype).tiny)
    lipschitz = largest**2 + plan.l2_regularization
    step = jnp.reciprocal(lipschitz)
    initial = jnp.full((points,), 0.0, dtype=design.dtype)
    if plan.sum_rule is not None:
        initial = jnp.full((points,), plan.sum_rule / points, dtype=design.dtype)

    def iteration(
        _: int, state: tuple[Array, Array, Array]
    ) -> tuple[Array, Array, Array]:
        value, extrapolated, momentum = state
        gradient = design.T @ (design @ extrapolated - target)
        gradient = gradient + plan.l2_regularization * extrapolated
        unconstrained = extrapolated - step * gradient
        if plan.sum_rule is not None:
            candidate = _project_simplex(unconstrained, plan.sum_rule)
        else:
            candidate = _soft_threshold(
                unconstrained,
                step * plan.l1_regularization,
            )
            if plan.nonnegative:
                candidate = jnp.maximum(candidate, 0.0)
        next_momentum = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * momentum**2))
        next_extrapolated = candidate + ((momentum - 1.0) / next_momentum) * (
            candidate - value
        )
        return candidate, next_extrapolated, next_momentum

    mass, _, _ = jax.lax.fori_loop(
        0,
        plan.maximum_iterations,
        iteration,
        (initial, initial, jnp.asarray(1.0, dtype=design.dtype)),
    )
    density = mass / plan.grid.weights
    residual_vector = design @ mass - target
    residual = jnp.sqrt(jnp.sum(residual_vector**2))
    target_norm = jnp.sqrt(jnp.sum(target**2))
    relative = residual / jnp.maximum(target_norm, jnp.finfo(target.dtype).tiny)
    regularization = plan.l1_regularization * jnp.sum(jnp.abs(mass))
    regularization = regularization + 0.5 * plan.l2_regularization * jnp.sum(mass**2)
    chi_square = jnp.sum(residual_vector**2)
    objective = 0.5 * chi_square + regularization
    smooth_gradient = design.T @ residual_vector + plan.l2_regularization * mass
    gradient_scale = jnp.maximum(
        1.0,
        jnp.max(jnp.abs(design.T @ target)),
    )
    if plan.sum_rule is None:
        prox = _soft_threshold(
            mass - step * smooth_gradient,
            step * plan.l1_regularization,
        )
        if plan.nonnegative:
            prox = jnp.maximum(prox, 0.0)
        gradient_norm = jnp.max(jnp.abs(mass - prox)) / step / gradient_scale
    else:
        active_kkt = mass > jnp.sqrt(jnp.finfo(mass.dtype).eps) * jnp.maximum(
            jnp.max(jnp.abs(mass)), 1.0
        )
        subgradient = smooth_gradient + plan.l1_regularization
        multiplier = jnp.sum(jnp.where(active_kkt, subgradient, 0.0)) / jnp.maximum(
            jnp.sum(active_kkt), 1
        )
        kkt_residual = jnp.where(
            active_kkt,
            jnp.abs(subgradient - multiplier),
            jnp.maximum(multiplier - subgradient, 0.0),
        )
        gradient_norm = jnp.max(kkt_residual) / gradient_scale
    active = jnp.abs(mass) > jnp.sqrt(jnp.finfo(mass.dtype).eps) * jnp.maximum(
        jnp.max(jnp.abs(mass)), 1.0
    )
    hessian = design.T @ design + plan.l2_regularization * jnp.eye(
        points, dtype=design.dtype
    )
    stabilization = max(
        plan.l2_regularization, float(np.finfo(np.dtype(design.dtype)).eps)
    )
    hessian = hessian + stabilization * jnp.diag((~active).astype(design.dtype))
    hessian_factor = _native_factor(hessian, 64 * np.finfo(np.dtype(hessian.dtype)).eps)
    covariance = hessian_factor.solve(jnp.eye(points, dtype=design.dtype)).value
    uncertainty_mass = jnp.sqrt(jnp.maximum(jnp.diag(covariance), 0.0))
    uncertainty = jnp.where(active, uncertainty_mass / plan.grid.weights, 0.0)
    singular_values = hessian_factor.singular_values()
    condition = singular_values[0] / jnp.maximum(
        singular_values[-1], jnp.finfo(singular_values.dtype).tiny
    )
    rank = jnp.sum(
        prepared.singular_values
        > 64 * jnp.finfo(design.dtype).eps * prepared.singular_values[0]
    )
    spectral_sum = jnp.sum(mass)
    sum_residual = (
        jnp.asarray(0.0, dtype=mass.dtype)
        if plan.sum_rule is None
        else jnp.abs(spectral_sum - plan.sum_rule)
    )
    finite_input = (
        prepared.samples.valid
        & jnp.all(jnp.isfinite(target))
        & jnp.all(jnp.isfinite(prepared.noise))
        & jnp.all(prepared.noise > 0.0)
    )
    finite = (
        finite_input
        & jnp.all(jnp.isfinite(density))
        & jnp.all(jnp.isfinite(uncertainty))
        & jnp.isfinite(objective)
    )
    positive = jnp.asarray(not plan.nonnegative) | (jnp.min(density) >= 0.0)
    expected_sum = 1.0 if plan.sum_rule is None else plan.sum_rule
    sum_valid = jnp.asarray(plan.sum_rule is None) | (
        sum_residual <= 64.0 * jnp.finfo(mass.dtype).eps * expected_sum
    )
    converged = gradient_norm <= plan.gradient_tolerance
    residual_valid = relative <= plan.residual_tolerance
    valid = finite & positive & sum_valid & residual_valid & converged
    status = jnp.where(
        ~finite_input,
        int(ContinuationStatus.NONFINITE_INPUT),
        jnp.where(
            ~finite,
            int(ContinuationStatus.NONFINITE_OUTPUT),
            jnp.where(
                ~positive,
                int(ContinuationStatus.POSITIVITY_VIOLATION),
                jnp.where(
                    ~sum_valid,
                    int(ContinuationStatus.SUM_RULE_VIOLATION),
                    jnp.where(
                        ~residual_valid,
                        int(ContinuationStatus.RESIDUAL_TOO_LARGE),
                        jnp.where(
                            converged,
                            int(ContinuationStatus.SUCCESS),
                            int(ContinuationStatus.MAXIMUM_ITERATIONS_REACHED),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    evidence = ContinuationEvidence(
        residual,
        relative,
        chi_square,
        regularization,
        objective,
        condition,
        rank.astype(jnp.int32),
        jnp.asarray(plan.maximum_iterations, dtype=jnp.int32),
        gradient_norm,
        jnp.min(density),
        spectral_sum,
        sum_residual,
        jnp.max(uncertainty),
        finite,
        converged,
        valid,
        status,
        "regularized-sparse",
    )
    return SpectralContinuationResult(
        plan.grid,
        density,
        uncertainty,
        evidence,
        _spectral_result_id("regularized-sparse", prepared.prepared_id),
    )


def sparse_continuation(
    samples: MatsubaraGreenFunction,
    grid: PreparedSpectralGrid,
    /,
    *,
    l1_regularization: float,
    l2_regularization: float = 0.0,
    nonnegative: bool = False,
    sum_rule: float | None = None,
    noise: ArrayLike = 1.0,
    maximum_iterations: int = 2000,
    gradient_tolerance: float = 1e-7,
    residual_tolerance: float = 1.0,
    maximum_bytes: int = 512 * 1024**2,
) -> SpectralContinuationResult:
    """Plan, prepare, and solve regularized sparse continuation."""

    plan = plan_sparse_continuation(
        grid,
        l1_regularization=l1_regularization,
        l2_regularization=l2_regularization,
        nonnegative=nonnegative,
        sum_rule=sum_rule,
        maximum_iterations=maximum_iterations,
        gradient_tolerance=gradient_tolerance,
        residual_tolerance=residual_tolerance,
        maximum_samples=samples.values.shape[0],
        maximum_bytes=maximum_bytes,
    )
    prepared = prepare_sparse_continuation(plan, samples, noise=noise)
    return solve_sparse_continuation(prepared)


class ScalarFermionicMaximumEntropyPlan(StrictModule):
    """Controlled scalar fermionic MaxEnt profile; matrix continuation is excluded."""

    inverse_problem: MaximumEntropyPlan
    expected_first_moment: float = eqx.field(static=True)
    moment_tolerance: float = eqx.field(static=True)
    frequency_unit: str = eqx.field(static=True)
    mode_label: str = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)


class PreparedScalarFermionicMaximumEntropy(StrictModule):
    plan: ScalarFermionicMaximumEntropyPlan
    inverse_problem: PreparedMaximumEntropy


class ScalarFermionicMaximumEntropyResult(StrictModule):
    inference: SpectralContinuationResult
    spectral: FermionicSpectralFunction
    profile_id: str = eqx.field(static=True)


def plan_scalar_fermionic_maximum_entropy(
    grid: PreparedSpectralGrid,
    /,
    *,
    alpha: float,
    expected_first_moment: float,
    sum_rule: float = 1.0,
    moment_tolerance: float = 1e-3,
    frequency_unit: str = "native-energy",
    mode_label: str = "local-orbital",
    maximum_iterations: int = 2000,
    gradient_tolerance: float = 1e-7,
    residual_tolerance: float = 1.0,
    learning_rate: float = 0.8,
    maximum_samples: int = 4096,
    maximum_bytes: int = 512 * 1024**2,
) -> ScalarFermionicMaximumEntropyPlan:
    first = float(expected_first_moment)
    tolerance = _positive(moment_tolerance, "moment_tolerance")
    unit = str(frequency_unit)
    mode = str(mode_label)
    if not isfinite(first) or not unit or not mode:
        raise ValueError("Scalar MaxEnt moment, unit, and mode metadata must be valid.")
    inverse = plan_maximum_entropy(
        grid,
        alpha=alpha,
        sum_rule=sum_rule,
        maximum_iterations=maximum_iterations,
        gradient_tolerance=gradient_tolerance,
        residual_tolerance=residual_tolerance,
        learning_rate=learning_rate,
        maximum_samples=maximum_samples,
        maximum_bytes=maximum_bytes,
    )
    profile_id = canonical_fingerprint(
        {
            "kind": "scalar-fermionic-maximum-entropy-profile",
            "inverse_problem": inverse.plan_id,
            "expected_first_moment": first,
            "moment_tolerance": tolerance,
            "frequency_unit": unit,
            "mode_label": mode,
        }
    )
    return ScalarFermionicMaximumEntropyPlan(
        inverse, first, tolerance, unit, mode, profile_id
    )


def prepare_scalar_fermionic_maximum_entropy(
    plan: ScalarFermionicMaximumEntropyPlan,
    samples: MatsubaraGreenFunction,
    /,
    *,
    noise: ArrayLike = 1.0,
    prior: ArrayLike | None = None,
) -> PreparedScalarFermionicMaximumEntropy:
    if not isinstance(plan, ScalarFermionicMaximumEntropyPlan):
        raise TypeError("plan must be ScalarFermionicMaximumEntropyPlan.")
    if samples.statistics != "fermionic" or samples.values.ndim != 1:
        raise ValueError(
            "The scalar fermionic MaxEnt profile excludes matrix/bosonic data."
        )
    prepared = prepare_maximum_entropy(
        plan.inverse_problem, samples, noise=noise, prior=prior
    )
    return PreparedScalarFermionicMaximumEntropy(plan, prepared)


def solve_scalar_fermionic_maximum_entropy(
    prepared: PreparedScalarFermionicMaximumEntropy, /
) -> ScalarFermionicMaximumEntropyResult:
    if not isinstance(prepared, PreparedScalarFermionicMaximumEntropy):
        raise TypeError("prepared must be PreparedScalarFermionicMaximumEntropy.")
    inference = solve_maximum_entropy(prepared.inverse_problem)
    plan = prepared.plan
    weights = inference.grid.weights
    density = inference.density
    positivity = jnp.max(jnp.maximum(-density, 0.0), initial=0.0)
    zeroth = contract("r,r->", weights, density)
    first = contract("r,r,r->", weights, inference.grid.frequencies, density)
    zeroth_residual = jnp.abs(zeroth - plan.inverse_problem.sum_rule)
    first_residual = jnp.abs(first - plan.expected_first_moment)
    finite = inference.evidence.finite & jnp.isfinite(first_residual)
    valid = (
        inference.evidence.valid
        & (positivity <= 0.0)
        & (zeroth_residual <= plan.moment_tolerance)
        & (first_residual <= plan.moment_tolerance)
    )
    status = jnp.where(
        valid,
        int(GreenFunctionStatus.SUCCESS),
        int(GreenFunctionStatus.RESIDUAL_TOO_LARGE),
    ).astype(jnp.int32)
    physicality = FermionicSpectralPhysicality(
        jnp.asarray(0.0, dtype=density.dtype),
        positivity,
        zeroth_residual,
        first_residual,
        finite,
        valid,
        status,
    )
    spectral = FermionicSpectralFunction(
        inference.grid.frequencies,
        density,
        weights,
        physicality,
        plan.frequency_unit,
        (plan.mode_label,),
        canonical_fingerprint(
            {
                "kind": "scalar-fermionic-maxent-spectrum",
                "profile": plan.profile_id,
                "inference": inference.continuation_id,
            }
        ),
    )
    return ScalarFermionicMaximumEntropyResult(inference, spectral, plan.profile_id)


def scalar_fermionic_maximum_entropy(
    samples: MatsubaraGreenFunction,
    grid: PreparedSpectralGrid,
    /,
    *,
    alpha: float,
    expected_first_moment: float,
    sum_rule: float = 1.0,
    noise: ArrayLike = 1.0,
    prior: ArrayLike | None = None,
    moment_tolerance: float = 1e-3,
    maximum_iterations: int = 2000,
    gradient_tolerance: float = 1e-7,
    residual_tolerance: float = 1.0,
    learning_rate: float = 0.8,
    maximum_bytes: int = 512 * 1024**2,
) -> ScalarFermionicMaximumEntropyResult:
    plan = plan_scalar_fermionic_maximum_entropy(
        grid,
        alpha=alpha,
        expected_first_moment=expected_first_moment,
        sum_rule=sum_rule,
        moment_tolerance=moment_tolerance,
        maximum_iterations=maximum_iterations,
        gradient_tolerance=gradient_tolerance,
        residual_tolerance=residual_tolerance,
        learning_rate=learning_rate,
        maximum_samples=samples.values.shape[0],
        maximum_bytes=maximum_bytes,
    )
    prepared = prepare_scalar_fermionic_maximum_entropy(
        plan, samples, noise=noise, prior=prior
    )
    return solve_scalar_fermionic_maximum_entropy(prepared)


__all__ = [
    "ContinuationEvidence",
    "ContinuationStatus",
    "MaximumEntropyPlan",
    "PadeContinuationPlan",
    "PadeEvaluation",
    "PreparedMaximumEntropy",
    "PreparedPadeContinuation",
    "PreparedSparseContinuation",
    "PreparedSpectralGrid",
    "PreparedScalarFermionicMaximumEntropy",
    "SparseContinuationPlan",
    "ScalarFermionicMaximumEntropyPlan",
    "ScalarFermionicMaximumEntropyResult",
    "SpectralContinuationResult",
    "SpectralGridPlan",
    "evaluate_pade",
    "evaluate_spectral_continuation",
    "maximum_entropy_continuation",
    "pade_continuation",
    "plan_maximum_entropy",
    "plan_pade_continuation",
    "plan_scalar_fermionic_maximum_entropy",
    "plan_sparse_continuation",
    "plan_spectral_grid",
    "prepare_maximum_entropy",
    "prepare_pade_continuation",
    "prepare_scalar_fermionic_maximum_entropy",
    "prepare_sparse_continuation",
    "prepare_spectral_grid",
    "solve_maximum_entropy",
    "scalar_fermionic_maximum_entropy",
    "solve_scalar_fermionic_maximum_entropy",
    "solve_sparse_continuation",
    "sparse_continuation",
    "spectral_grid",
]
