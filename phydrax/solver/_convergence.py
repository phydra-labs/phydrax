#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import isfinite, sqrt
from statistics import NormalDist
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._frozendict import frozendict
from .._strict import StrictModule


SPDERefinementAxis: TypeAlias = Literal[
    "time",
    "space",
    "noise_rank",
    "ensemble",
]
SPDEConvergenceMetric: TypeAlias = Literal[
    "strong",
    "pathwise",
    "temporal",
    "spatial",
    "noise",
    "sampling",
    "invariant",
]


def _optional_error(value: float | None, name: str, /) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative or None.")
    return result


class WeakObservableEstimate(StrictModule):
    """Weak observable estimate with a normal-approximation confidence interval."""

    estimate: float = eqx.field(static=True)
    reference: float = eqx.field(static=True)
    standard_error: float = eqx.field(static=True)
    confidence_level: float = eqx.field(static=True)
    sample_size: int = eqx.field(static=True)
    name: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        estimate: float,
        reference: float,
        standard_error: float,
        sample_size: int,
        /,
        *,
        confidence_level: float = 0.95,
    ):
        resolved_name = str(name)
        values = (float(estimate), float(reference), float(standard_error))
        if not resolved_name:
            raise ValueError("Weak observable names must be non-empty.")
        if any(not isfinite(value) for value in values) or values[2] < 0.0:
            raise ValueError("Weak estimates and references must be finite with SE >= 0.")
        count = int(sample_size)
        if count <= 0:
            raise ValueError("sample_size must be positive.")
        confidence = float(confidence_level)
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one.")
        self.name = resolved_name
        self.estimate, self.reference, self.standard_error = values
        self.sample_size = count
        self.confidence_level = confidence

    @property
    def error(self) -> float:
        return abs(self.estimate - self.reference)

    @property
    def confidence_interval(self) -> tuple[float, float]:
        quantile = NormalDist().inv_cdf(0.5 + 0.5 * self.confidence_level)
        radius = quantile * self.standard_error
        return self.estimate - radius, self.estimate + radius

    @property
    def reference_covered(self) -> bool:
        lower, upper = self.confidence_interval
        return lower <= self.reference <= upper


class SPDEErrorBudget(StrictModule):
    """Orthogonal temporal, spatial, noise, and sampling error accounting."""

    temporal: float = eqx.field(static=True)
    spatial: float = eqx.field(static=True)
    noise: float = eqx.field(static=True)
    sampling: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        temporal: float = 0.0,
        spatial: float = 0.0,
        noise: float = 0.0,
        sampling: float = 0.0,
    ):
        values = tuple(float(value) for value in (temporal, spatial, noise, sampling))
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "Every SPDE error-budget term must be finite and non-negative."
            )
        self.temporal, self.spatial, self.noise, self.sampling = values

    @property
    def total_upper_bound(self) -> float:
        return self.temporal + self.spatial + self.noise + self.sampling

    def component(self, axis: SPDERefinementAxis, /) -> float:
        if axis == "time":
            return self.temporal
        if axis == "space":
            return self.spatial
        if axis == "noise_rank":
            return self.noise
        if axis == "ensemble":
            return self.sampling
        raise ValueError(f"Unknown refinement axis {axis!r}.")


class SPDEConvergenceLevel(StrictModule):
    """One refinement level with strong, weak, stability, and provenance data."""

    weak_estimates: tuple[WeakObservableEstimate, ...]
    error_budget: SPDEErrorBudget | None
    provenance: frozendict[str, str]
    resolution: float = eqx.field(static=True)
    work: float = eqx.field(static=True)
    strong_error: float | None = eqx.field(static=True)
    pathwise_error: float | None = eqx.field(static=True)
    mean_square: float | None = eqx.field(static=True)
    invariant_error: float | None = eqx.field(static=True)
    realization_id: str | None = eqx.field(static=True)
    coupling_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        resolution: float,
        /,
        *,
        work: float,
        strong_error: float | None = None,
        pathwise_error: float | None = None,
        weak_estimates: Sequence[WeakObservableEstimate] = (),
        error_budget: SPDEErrorBudget | None = None,
        mean_square: float | None = None,
        invariant_error: float | None = None,
        realization_id: str | None = None,
        coupling_id: str | None = None,
        provenance: Mapping[str, str] | None = None,
    ):
        scale = float(resolution)
        cost = float(work)
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError("resolution must be finite and positive.")
        if not isfinite(cost) or cost <= 0.0:
            raise ValueError("work must be finite and positive.")
        weak = tuple(weak_estimates)
        if any(not isinstance(value, WeakObservableEstimate) for value in weak):
            raise TypeError("weak_estimates must contain WeakObservableEstimate objects.")
        names = tuple(value.name for value in weak)
        if len(set(names)) != len(names):
            raise ValueError("Weak observable names must be unique within a level.")
        if error_budget is not None and not isinstance(error_budget, SPDEErrorBudget):
            raise TypeError("error_budget must be SPDEErrorBudget or None.")
        identities = frozendict(
            {} if provenance is None else {str(k): str(v) for k, v in provenance.items()}
        )
        if any(not key or not value for key, value in identities.items()):
            raise ValueError("Convergence provenance keys and values must be non-empty.")
        for name, value in (
            ("realization_id", realization_id),
            ("coupling_id", coupling_id),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be non-empty or None.")
        self.resolution = scale
        self.work = cost
        self.strong_error = _optional_error(strong_error, "strong_error")
        self.pathwise_error = _optional_error(pathwise_error, "pathwise_error")
        self.weak_estimates = weak
        self.error_budget = error_budget
        self.mean_square = _optional_error(mean_square, "mean_square")
        self.invariant_error = _optional_error(invariant_error, "invariant_error")
        self.realization_id = realization_id
        self.coupling_id = coupling_id
        self.provenance = identities

    def weak(self, name: str, /) -> WeakObservableEstimate:
        matches = tuple(value for value in self.weak_estimates if value.name == name)
        if len(matches) != 1:
            raise KeyError(f"Unknown weak observable {name!r}.")
        return matches[0]

    def metric(self, metric: SPDEConvergenceMetric, /) -> float:
        if metric == "strong":
            value = self.strong_error
        elif metric == "pathwise":
            value = self.pathwise_error
        elif metric == "invariant":
            value = self.invariant_error
        elif self.error_budget is not None:
            mapping = {
                "temporal": self.error_budget.temporal,
                "spatial": self.error_budget.spatial,
                "noise": self.error_budget.noise,
                "sampling": self.error_budget.sampling,
            }
            value = mapping[metric]
        else:
            value = None
        if value is None:
            raise ValueError(
                f"Metric {metric!r} is absent at resolution {self.resolution}."
            )
        return float(value)


class SPDEConvergenceStudy(StrictModule):
    """One-axis-at-a-time SPDE convergence study with auditable coupling."""

    levels: tuple[SPDEConvergenceLevel, ...]
    refined_axis: SPDERefinementAxis = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)

    def __init__(
        self,
        refined_axis: SPDERefinementAxis,
        levels: Sequence[SPDEConvergenceLevel],
        /,
        *,
        reference_id: str,
    ):
        if refined_axis not in ("time", "space", "noise_rank", "ensemble"):
            raise ValueError(f"Unknown SPDE refinement axis {refined_axis!r}.")
        values = tuple(levels)
        if len(values) < 2 or any(
            not isinstance(value, SPDEConvergenceLevel) for value in values
        ):
            raise ValueError("A convergence study requires at least two valid levels.")
        values = tuple(sorted(values, key=lambda value: value.resolution, reverse=True))
        resolutions = tuple(value.resolution for value in values)
        if len(set(resolutions)) != len(resolutions):
            raise ValueError("Convergence resolutions must be unique.")
        identifier = str(reference_id)
        if not identifier:
            raise ValueError("reference_id must be non-empty.")
        coupled = tuple(
            level
            for level in values
            if level.strong_error is not None or level.pathwise_error is not None
        )
        if coupled:
            coupling_ids = {level.coupling_id for level in coupled}
            if None in coupling_ids or len(coupling_ids) != 1:
                raise ValueError(
                    "All strong/pathwise levels must share one explicit coupling_id."
                )
        self.refined_axis = refined_axis
        self.levels = values
        self.reference_id = identifier

    @property
    def resolutions(self) -> Array:
        return jnp.asarray(tuple(level.resolution for level in self.levels))

    def errors(
        self,
        metric: SPDEConvergenceMetric = "strong",
        /,
        *,
        observable: str | None = None,
    ) -> Array:
        if observable is not None:
            return jnp.asarray(
                tuple(level.weak(observable).error for level in self.levels)
            )
        return jnp.asarray(tuple(level.metric(metric) for level in self.levels))

    def pairwise_rates(
        self,
        metric: SPDEConvergenceMetric = "strong",
        /,
        *,
        observable: str | None = None,
    ) -> Array:
        errors = self.errors(metric, observable=observable)
        if bool(jnp.any(errors <= 0.0)):
            raise ValueError("Empirical rates require strictly positive errors.")
        scales = self.resolutions
        return jnp.log(errors[:-1] / errors[1:]) / jnp.log(scales[:-1] / scales[1:])

    def regression_rate(
        self,
        metric: SPDEConvergenceMetric = "strong",
        /,
        *,
        observable: str | None = None,
    ) -> float:
        errors = np.asarray(self.errors(metric, observable=observable), dtype=float)
        if np.any(errors <= 0.0):
            raise ValueError("Empirical rates require strictly positive errors.")
        slope = np.polyfit(np.log(np.asarray(self.resolutions)), np.log(errors), 1)[0]
        return float(slope)

    def sampling_is_subordinate(
        self,
        metric: SPDEConvergenceMetric = "strong",
        /,
        *,
        factor: float = 1.0,
        observable: str | None = None,
    ) -> bool:
        threshold = float(factor)
        if not isfinite(threshold) or threshold <= 0.0:
            raise ValueError("factor must be finite and positive.")
        errors = np.asarray(self.errors(metric, observable=observable))
        sampling = np.asarray(
            tuple(
                np.inf if level.error_budget is None else level.error_budget.sampling
                for level in self.levels
            )
        )
        return bool(np.all(sampling <= threshold * errors))

    def mean_square_stable(self, /, *, upper_bound: float | None = None) -> bool:
        values = tuple(level.mean_square for level in self.levels)
        if any(value is None for value in values):
            return False
        bound = np.inf if upper_bound is None else float(upper_bound)
        if not isfinite(bound) and upper_bound is not None:
            raise ValueError("upper_bound must be finite when provided.")
        return bool(np.all(np.asarray(values, dtype=float) <= bound))


def weak_observable_estimate(
    samples: ArrayLike,
    observable: Callable[[Array], ArrayLike],
    reference: float,
    /,
    *,
    name: str,
    confidence_level: float = 0.95,
) -> WeakObservableEstimate:
    """Estimate an arbitrary scalar weak observable over a leading sample axis."""
    if not callable(observable):
        raise TypeError("observable must be callable.")
    values = jnp.asarray(samples)
    if values.ndim < 1 or int(values.shape[0]) < 2:
        raise ValueError("Weak estimates require at least two leading samples.")
    evaluated = jnp.asarray(
        tuple(observable(values[index]) for index in range(int(values.shape[0]))),
        dtype=float,
    ).reshape((-1,))
    if evaluated.shape != (int(values.shape[0]),):
        raise ValueError("observable must return one scalar per sample.")
    estimate = float(jnp.mean(evaluated))
    standard_error = float(jnp.std(evaluated, ddof=1) / jnp.sqrt(float(evaluated.size)))
    return WeakObservableEstimate(
        name,
        estimate,
        float(reference),
        standard_error,
        int(evaluated.size),
        confidence_level=confidence_level,
    )


def coupled_strong_error(
    approximation: ArrayLike,
    reference: ArrayLike,
    /,
    *,
    quadrature_weights: ArrayLike | None = None,
) -> float:
    """Root mean-square state error for samples sharing one global realization."""
    left = jnp.asarray(approximation)
    right = jnp.asarray(reference)
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("Coupled strong arrays must share sample-first state shape.")
    difference = left - right
    if quadrature_weights is None:
        squared = jnp.mean(jnp.abs(difference) ** 2, axis=tuple(range(1, left.ndim)))
    else:
        weights = jnp.asarray(quadrature_weights, dtype=float)
        if difference.shape[1 : 1 + weights.ndim] != weights.shape:
            raise ValueError("quadrature_weights must match leading state dimensions.")
        broadcast = weights.reshape(
            (1,) + weights.shape + (1,) * (left.ndim - 1 - weights.ndim)
        )
        numerator = jnp.sum(
            broadcast * jnp.abs(difference) ** 2,
            axis=tuple(range(1, left.ndim)),
        )
        denominator = jnp.sum(weights) * float(
            np.prod(left.shape[1 + weights.ndim :]) if left.ndim > 1 + weights.ndim else 1
        )
        squared = numerator / denominator
    return float(jnp.sqrt(jnp.mean(squared)))


class NoiseTruncationLevel(StrictModule):
    """Raw and solution-aware errors for one retained spatial noise rank."""

    weak_observable_residuals: frozendict[str, float]
    rank: int = eqx.field(static=True)
    raw_covariance_residual: float = eqx.field(static=True)
    relative_raw_covariance_residual: float = eqx.field(static=True)
    finite_horizon_solution_residual: float = eqx.field(static=True)
    relative_finite_horizon_residual: float = eqx.field(static=True)
    stationary_solution_residual: float | None = eqx.field(static=True)
    strong_rms_error: float = eqx.field(static=True)

    def __init__(
        self,
        rank: int,
        /,
        *,
        raw_covariance_residual: float,
        relative_raw_covariance_residual: float,
        finite_horizon_solution_residual: float,
        relative_finite_horizon_residual: float,
        stationary_solution_residual: float | None,
        strong_rms_error: float,
        weak_observable_residuals: Mapping[str, float] | None = None,
    ):
        retained = int(rank)
        if retained < 0:
            raise ValueError("rank must be non-negative.")
        values = (
            raw_covariance_residual,
            relative_raw_covariance_residual,
            finite_horizon_solution_residual,
            relative_finite_horizon_residual,
            strong_rms_error,
        )
        if any(not isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise ValueError(
                "Noise truncation residuals must be finite and non-negative."
            )
        stationary = _optional_error(
            stationary_solution_residual,
            "stationary_solution_residual",
        )
        weak = frozendict(
            {}
            if weak_observable_residuals is None
            else {
                str(name): float(value)
                for name, value in weak_observable_residuals.items()
            }
        )
        if any(
            not name or not isfinite(value) or value < 0.0 for name, value in weak.items()
        ):
            raise ValueError("Weak truncation residuals must be named and non-negative.")
        self.rank = retained
        self.raw_covariance_residual = float(raw_covariance_residual)
        self.relative_raw_covariance_residual = float(relative_raw_covariance_residual)
        self.finite_horizon_solution_residual = float(finite_horizon_solution_residual)
        self.relative_finite_horizon_residual = float(relative_finite_horizon_residual)
        self.stationary_solution_residual = stationary
        self.strong_rms_error = float(strong_rms_error)
        self.weak_observable_residuals = weak


class NoiseTruncationStudy(StrictModule):
    """Operator-aware finite-rank noise diagnostics outside SpatialNoiseBasis."""

    levels: tuple[NoiseTruncationLevel, ...]
    horizon: float = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        levels: Sequence[NoiseTruncationLevel],
        /,
        *,
        horizon: float,
        operator_id: str,
        basis_id: str,
    ):
        values = tuple(levels)
        if not values or any(
            not isinstance(value, NoiseTruncationLevel) for value in values
        ):
            raise ValueError("NoiseTruncationStudy requires at least one valid level.")
        ranks = tuple(value.rank for value in values)
        if tuple(sorted(ranks)) != ranks or len(set(ranks)) != len(ranks):
            raise ValueError("Noise truncation ranks must be unique and increasing.")
        time = float(horizon)
        if not isfinite(time) or time <= 0.0:
            raise ValueError("horizon must be finite and positive.")
        if not str(operator_id) or not str(basis_id):
            raise ValueError("operator_id and basis_id must be non-empty.")
        self.levels = values
        self.horizon = time
        self.operator_id = str(operator_id)
        self.basis_id = str(basis_id)

    @classmethod
    def from_compatible_spectrum(
        cls,
        covariance_eigenvalues: ArrayLike,
        linear_eigenvalues: ArrayLike,
        ranks: Sequence[int],
        /,
        *,
        horizon: float,
        operator_id: str,
        basis_id: str,
        observable_mode_weights: Mapping[str, ArrayLike] | None = None,
    ) -> "NoiseTruncationStudy":
        covariance = np.asarray(covariance_eigenvalues, dtype=float).reshape((-1,))
        linear = np.asarray(linear_eigenvalues, dtype=float).reshape((-1,))
        if covariance.shape != linear.shape or covariance.size <= 0:
            raise ValueError(
                "Covariance and linear spectra must be equal non-empty vectors."
            )
        if np.any(~np.isfinite(covariance)) or np.any(covariance < 0.0):
            raise ValueError("Covariance eigenvalues must be finite and non-negative.")
        if np.any(~np.isfinite(linear)):
            raise ValueError("Linear eigenvalues must be finite.")
        retained_ranks = tuple(int(rank) for rank in ranks)
        if any(rank < 0 or rank > covariance.size for rank in retained_ranks):
            raise ValueError("Every retained rank must lie in [0, spectrum size].")
        time = float(horizon)
        if not isfinite(time) or time <= 0.0:
            raise ValueError("horizon must be finite and positive.")
        threshold = np.sqrt(np.finfo(float).eps)
        factors = np.where(
            np.abs(linear) > threshold,
            np.expm1(2.0 * linear * time)
            / (2.0 * np.where(np.abs(linear) > threshold, linear, 1.0)),
            time + linear * time**2 + (2.0 / 3.0) * linear**2 * time**3,
        )
        solution_energy = covariance * factors
        raw_total = float(np.sum(covariance))
        solution_total = float(np.sum(solution_energy))
        weights: dict[str, np.ndarray] = {}
        for name, value in (
            {} if observable_mode_weights is None else observable_mode_weights
        ).items():
            resolved = np.asarray(value, dtype=float).reshape((-1,))
            if resolved.shape != covariance.shape or np.any(~np.isfinite(resolved)):
                raise ValueError(
                    "Every observable mode weight must be finite and match the spectra."
                )
            weights[str(name)] = resolved
        levels: list[NoiseTruncationLevel] = []
        for rank in retained_ranks:
            raw = float(np.sum(covariance[rank:]))
            finite = float(np.sum(solution_energy[rank:]))
            omitted_active = covariance[rank:] > 0.0
            if np.any(omitted_active & (linear[rank:] >= 0.0)):
                stationary = None
            else:
                active_covariance = covariance[rank:][omitted_active]
                active_linear = linear[rank:][omitted_active]
                stationary = float(np.sum(active_covariance / (-2.0 * active_linear)))
            weak = {
                name: sqrt(
                    max(
                        0.0,
                        float(np.sum(solution_energy[rank:] * value[rank:] ** 2)),
                    )
                )
                for name, value in weights.items()
            }
            levels.append(
                NoiseTruncationLevel(
                    rank,
                    raw_covariance_residual=raw,
                    relative_raw_covariance_residual=(
                        0.0 if raw_total == 0.0 else raw / raw_total
                    ),
                    finite_horizon_solution_residual=finite,
                    relative_finite_horizon_residual=(
                        0.0 if solution_total == 0.0 else finite / solution_total
                    ),
                    stationary_solution_residual=stationary,
                    strong_rms_error=sqrt(max(0.0, finite)),
                    weak_observable_residuals=weak,
                )
            )
        return cls(
            levels,
            horizon=time,
            operator_id=operator_id,
            basis_id=basis_id,
        )

    def recommended_rank(
        self,
        tolerance: float,
        /,
        *,
        metric: Literal[
            "raw", "finite_horizon", "strong_rms", "stationary"
        ] = "finite_horizon",
    ) -> int:
        threshold = float(tolerance)
        if not isfinite(threshold) or threshold < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        for level in self.levels:
            if metric == "raw":
                value = level.raw_covariance_residual
            elif metric == "finite_horizon":
                value = level.finite_horizon_solution_residual
            elif metric == "strong_rms":
                value = level.strong_rms_error
            elif metric == "stationary":
                value = level.stationary_solution_residual
                if value is None:
                    continue
            else:
                raise ValueError(f"Unknown truncation metric {metric!r}.")
            if value <= threshold:
                return level.rank
        raise ValueError(f"No studied rank reaches {metric!r} tolerance {threshold:.6g}.")


__all__ = [
    "NoiseTruncationLevel",
    "NoiseTruncationStudy",
    "SPDEConvergenceLevel",
    "SPDEConvergenceMetric",
    "SPDEConvergenceStudy",
    "SPDEErrorBudget",
    "SPDERefinementAxis",
    "WeakObservableEstimate",
    "coupled_strong_error",
    "weak_observable_estimate",
]
