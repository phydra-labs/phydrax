#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...linalg import (
    DenseLinearOperator,
    DenseSVD,
    LinearSolvePolicy,
    LinearSystem,
    prepare,
    solve,
)


def _positive(value: float, name: str, /, *, allow_zero: bool = False) -> float:
    result = float(value)
    valid = result >= 0.0 if allow_zero else result > 0.0
    if not isfinite(result) or not valid:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}.")
    return result


def _probability(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one.")
    return result


def _vector(value: ArrayLike, name: str, /, *, nonempty: bool = True) -> Array:
    result = jnp.asarray(value)
    if result.ndim != 1 or (nonempty and result.shape[0] == 0):
        raise ValueError(f"{name} must be a{' non-empty' if nonempty else ''} vector.")
    if not jnp.issubdtype(result.dtype, jnp.inexact) or jnp.issubdtype(
        result.dtype, jnp.complexfloating
    ):
        result = result.astype(jnp.result_type(result.dtype, jnp.float32))
    if not bool(np.all(np.isfinite(np.asarray(result)))):
        raise ValueError(f"{name} must be finite.")
    return result


def _matrix(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value)
    if result.ndim != 2 or 0 in result.shape:
        raise ValueError(f"{name} must be a non-empty matrix.")
    if not jnp.issubdtype(result.dtype, jnp.inexact) or jnp.issubdtype(
        result.dtype, jnp.complexfloating
    ):
        result = result.astype(jnp.result_type(result.dtype, jnp.float32))
    if not bool(np.all(np.isfinite(np.asarray(result)))):
        raise ValueError(f"{name} must be finite.")
    return result


class MeanVarianceObjective(StrictModule):
    """Expected-return reward with a positive-semidefinite variance penalty."""

    risk_aversion: float = eqx.field(static=True)
    return_weight: float = eqx.field(static=True)

    def __init__(self, risk_aversion: float, /, *, return_weight: float = 1.0):
        self.risk_aversion = _positive(risk_aversion, "risk_aversion", allow_zero=True)
        self.return_weight = _positive(return_weight, "return_weight", allow_zero=True)
        if self.risk_aversion == 0.0 and self.return_weight == 0.0:
            raise ValueError("At least one objective weight must be positive.")


class TrackingErrorObjective(StrictModule):
    """Quadratic active-risk penalty around an explicit benchmark."""

    benchmark_weights: Array
    tracking_aversion: float = eqx.field(static=True)
    return_weight: float = eqx.field(static=True)

    def __init__(
        self,
        benchmark_weights: ArrayLike,
        /,
        *,
        tracking_aversion: float = 1.0,
        return_weight: float = 0.0,
    ):
        self.benchmark_weights = _vector(benchmark_weights, "benchmark_weights")
        self.tracking_aversion = _positive(tracking_aversion, "tracking_aversion")
        self.return_weight = _positive(return_weight, "return_weight", allow_zero=True)


class BlackLittermanObjective(StrictModule):
    """Mean-variance allocation under an explicit Black--Litterman view law."""

    equilibrium_returns: Array
    pick_matrix: Array
    view_returns: Array
    view_covariance: Array
    tau: float = eqx.field(static=True)
    risk_aversion: float = eqx.field(static=True)

    def __init__(
        self,
        equilibrium_returns: ArrayLike,
        pick_matrix: ArrayLike,
        view_returns: ArrayLike,
        view_covariance: ArrayLike,
        /,
        *,
        tau: float = 0.05,
        risk_aversion: float = 1.0,
    ):
        equilibrium = _vector(equilibrium_returns, "equilibrium_returns")
        pick = _matrix(pick_matrix, "pick_matrix")
        views = _vector(view_returns, "view_returns")
        covariance = _matrix(view_covariance, "view_covariance")
        assets = int(equilibrium.shape[0])
        count = int(views.shape[0])
        if pick.shape != (count, assets):
            raise ValueError(
                f"pick_matrix must have shape ({count}, {assets}); got {pick.shape}."
            )
        if covariance.shape != (count, count):
            raise ValueError(
                f"view_covariance must have shape ({count}, {count}); got {covariance.shape}."
            )
        symmetric = 0.5 * np.asarray(covariance) + 0.5 * np.asarray(covariance).T
        tolerance = (
            64.0
            * np.finfo(symmetric.dtype).eps
            * max(float(np.max(np.abs(symmetric))), 1.0)
        )
        if float(np.min(np.linalg.eigvalsh(symmetric))) < -tolerance:
            raise ValueError("view_covariance must be positive semidefinite.")
        self.equilibrium_returns = equilibrium
        self.pick_matrix = pick.astype(equilibrium.dtype)
        self.view_returns = views.astype(equilibrium.dtype)
        self.view_covariance = jnp.asarray(symmetric, dtype=equilibrium.dtype)
        self.tau = _positive(tau, "tau")
        self.risk_aversion = _positive(risk_aversion, "risk_aversion")

    def posterior(self, covariance: ArrayLike, /) -> tuple[Array, Array]:
        """Return posterior mean and covariance using the shared rank-safe solver."""

        prior = _matrix(covariance, "covariance").astype(self.equilibrium_returns.dtype)
        assets = int(self.equilibrium_returns.shape[0])
        if prior.shape != (assets, assets):
            raise ValueError(
                f"covariance must have shape ({assets}, {assets}); got {prior.shape}."
            )
        prior = 0.5 * (prior + prior.T)
        scaled = self.tau * prior
        innovation_covariance = (
            self.pick_matrix @ scaled @ self.pick_matrix.T + self.view_covariance
        )
        innovation = self.view_returns - self.pick_matrix @ self.equilibrium_returns
        prepared = prepare(
            LinearSystem(DenseLinearOperator(innovation_covariance)),
            LinearSolvePolicy(DenseSVD()),
        )
        solved = solve(prepared, innovation).value
        posterior_mean = self.equilibrium_returns + scaled @ self.pick_matrix.T @ solved
        gain_rhs = self.pick_matrix @ scaled
        solved_covariance = solve(prepared, gain_rhs).value
        posterior_covariance = (
            prior + scaled - scaled @ self.pick_matrix.T @ solved_covariance
        )
        return posterior_mean, 0.5 * (posterior_covariance + posterior_covariance.T)


class FiniteScenarioKellyObjective(StrictModule):
    """Expected logarithmic growth over a finite return law."""

    initial_wealth: float = eqx.field(static=True)
    bankruptcy_floor: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_wealth: float = 1.0,
        bankruptcy_floor: float = 1e-8,
    ):
        wealth = _positive(initial_wealth, "initial_wealth")
        floor = _positive(bankruptcy_floor, "bankruptcy_floor")
        if floor >= wealth:
            raise ValueError("bankruptcy_floor must be below initial_wealth.")
        self.initial_wealth = wealth
        self.bankruptcy_floor = floor


class CVaRObjective(StrictModule):
    """Expected shortfall of finite-scenario losses with optional return reward."""

    confidence: float = eqx.field(static=True)
    risk_weight: float = eqx.field(static=True)
    return_weight: float = eqx.field(static=True)

    def __init__(
        self,
        confidence: float,
        /,
        *,
        risk_weight: float = 1.0,
        return_weight: float = 0.0,
    ):
        self.confidence = _probability(confidence, "confidence")
        self.risk_weight = _positive(risk_weight, "risk_weight")
        self.return_weight = _positive(return_weight, "return_weight", allow_zero=True)


class EVaRObjective(StrictModule):
    """Entropic value-at-risk represented by exponential-cone perspectives."""

    confidence: float = eqx.field(static=True)
    risk_weight: float = eqx.field(static=True)
    return_weight: float = eqx.field(static=True)

    def __init__(
        self,
        confidence: float,
        /,
        *,
        risk_weight: float = 1.0,
        return_weight: float = 0.0,
    ):
        self.confidence = _probability(confidence, "confidence")
        self.risk_weight = _positive(risk_weight, "risk_weight")
        self.return_weight = _positive(return_weight, "return_weight", allow_zero=True)

    @property
    def relative_entropy_radius(self) -> float:
        return -float(np.log1p(-self.confidence))


class KLDivergenceRobustObjective(StrictModule):
    """Worst-case expected loss over a KL ball around scenario probabilities."""

    radius: float = eqx.field(static=True)
    risk_weight: float = eqx.field(static=True)
    return_weight: float = eqx.field(static=True)

    def __init__(
        self,
        radius: float,
        /,
        *,
        risk_weight: float = 1.0,
        return_weight: float = 0.0,
    ):
        self.radius = _positive(radius, "radius")
        self.risk_weight = _positive(risk_weight, "risk_weight")
        self.return_weight = _positive(return_weight, "return_weight", allow_zero=True)


class SpectralRiskObjective(StrictModule):
    """Positive mixture of expected-shortfall atoms."""

    confidences: Array
    weights: Array
    risk_weight: float = eqx.field(static=True)
    return_weight: float = eqx.field(static=True)

    def __init__(
        self,
        confidences: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        risk_weight: float = 1.0,
        return_weight: float = 0.0,
    ):
        levels = _vector(confidences, "confidences")
        masses = _vector(weights, "weights").astype(levels.dtype)
        if masses.shape != levels.shape:
            raise ValueError("confidences and weights must have identical shapes.")
        if bool(np.any((np.asarray(levels) <= 0.0) | (np.asarray(levels) >= 1.0))):
            raise ValueError("Every spectral confidence must lie in (0, 1).")
        if bool(np.any(np.asarray(masses) < 0.0)):
            raise ValueError("Spectral weights must be non-negative.")
        total = float(np.sum(np.asarray(masses)))
        tolerance = 64.0 * np.finfo(np.asarray(masses).dtype).eps
        if abs(total - 1.0) > tolerance:
            raise ValueError("Spectral weights must sum to one.")
        self.confidences = levels
        self.weights = masses
        self.risk_weight = _positive(risk_weight, "risk_weight")
        self.return_weight = _positive(return_weight, "return_weight", allow_zero=True)


class DrawdownRiskObjective(StrictModule):
    """Expected maximum additive drawdown over finite scenario paths."""

    risk_weight: float = eqx.field(static=True)
    return_weight: float = eqx.field(static=True)

    def __init__(self, *, risk_weight: float = 1.0, return_weight: float = 0.0):
        self.risk_weight = _positive(risk_weight, "risk_weight")
        self.return_weight = _positive(return_weight, "return_weight", allow_zero=True)


PortfolioObjective: TypeAlias = (
    MeanVarianceObjective
    | TrackingErrorObjective
    | BlackLittermanObjective
    | FiniteScenarioKellyObjective
    | CVaRObjective
    | EVaRObjective
    | KLDivergenceRobustObjective
    | SpectralRiskObjective
    | DrawdownRiskObjective
)


__all__ = [
    "BlackLittermanObjective",
    "CVaRObjective",
    "DrawdownRiskObjective",
    "EVaRObjective",
    "FiniteScenarioKellyObjective",
    "KLDivergenceRobustObjective",
    "MeanVarianceObjective",
    "PortfolioObjective",
    "SpectralRiskObjective",
    "TrackingErrorObjective",
]
