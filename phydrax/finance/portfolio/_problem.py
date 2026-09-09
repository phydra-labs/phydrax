#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..core import FinanceEvidenceBinding, PhysicalLaw
from ._constraints import PortfolioConstraints
from ._objectives import (
    BlackLittermanObjective,
    CVaRObjective,
    DrawdownRiskObjective,
    EVaRObjective,
    FiniteScenarioKellyObjective,
    KLDivergenceRobustObjective,
    MeanVarianceObjective,
    PortfolioObjective,
    SpectralRiskObjective,
    TrackingErrorObjective,
)


def _real_array(value: ArrayLike, name: str, /, *, ndim: int) -> Array:
    result = jnp.asarray(value)
    if result.ndim != ndim or 0 in result.shape:
        raise ValueError(f"{name} must be a non-empty rank-{ndim} array.")
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    result = result.astype(jnp.result_type(result.dtype, jnp.float32))
    if not bool(np.all(np.isfinite(np.asarray(result)))):
        raise ValueError(f"{name} must be finite.")
    return result


class ForecastLaw(StrictModule):
    """Decision-time portfolio forecast, separate from solve and realization evidence."""

    expected_returns: Array
    covariance: Array
    scenario_returns: Array | None
    scenario_probabilities: Array | None
    evidence: FinanceEvidenceBinding
    law: PhysicalLaw = eqx.field(static=True)
    asset_ids: tuple[str, ...] = eqx.field(static=True)
    as_of_time_ns: int = eqx.field(static=True)
    available_time_ns: int = eqx.field(static=True)

    def __init__(
        self,
        asset_ids: tuple[str, ...],
        expected_returns: ArrayLike,
        covariance: ArrayLike,
        /,
        *,
        law: PhysicalLaw,
        as_of_time_ns: int,
        available_time_ns: int,
        evidence: FinanceEvidenceBinding,
        scenario_returns: ArrayLike | None = None,
        scenario_probabilities: ArrayLike | None = None,
    ):
        assets = tuple(str(asset) for asset in asset_ids)
        if (
            not assets
            or any(not asset for asset in assets)
            or len(set(assets)) != len(assets)
        ):
            raise ValueError("asset_ids must be non-empty and unique.")
        expected = _real_array(expected_returns, "expected_returns", ndim=1)
        covariance_ = _real_array(covariance, "covariance", ndim=2).astype(expected.dtype)
        count = len(assets)
        if expected.shape != (count,) or covariance_.shape != (count, count):
            raise ValueError(
                f"Forecast moments must have shapes ({count},) and ({count}, {count})."
            )
        covariance_host = np.asarray(covariance_)
        symmetry_tolerance = (
            64.0
            * np.finfo(covariance_host.dtype).eps
            * max(float(np.max(np.abs(covariance_host))), 1.0)
        )
        if np.max(np.abs(covariance_host - covariance_host.T)) > symmetry_tolerance:
            raise ValueError("covariance must be symmetric.")
        symmetric = 0.5 * covariance_host + 0.5 * covariance_host.T
        if float(np.min(np.linalg.eigvalsh(symmetric))) < -symmetry_tolerance:
            raise ValueError(
                "covariance must be positive semidefinite; singular is allowed."
            )
        supplied_scenarios = scenario_returns is not None
        if supplied_scenarios != (scenario_probabilities is not None):
            raise ValueError(
                "scenario_returns and scenario_probabilities must be supplied together."
            )
        scenarios = None
        probabilities = None
        if supplied_scenarios:
            scenarios = jnp.asarray(scenario_returns, dtype=expected.dtype)
            if scenarios.ndim not in (2, 3) or scenarios.shape[-1] != count:
                raise ValueError(
                    "scenario_returns must have shape (scenario, asset) or "
                    "(scenario, stage, asset)."
                )
            if 0 in scenarios.shape or not bool(
                np.all(np.isfinite(np.asarray(scenarios)))
            ):
                raise ValueError("scenario_returns must be non-empty and finite.")
            probabilities = _real_array(
                scenario_probabilities, "scenario_probabilities", ndim=1
            ).astype(expected.dtype)
            if probabilities.shape != (scenarios.shape[0],):
                raise ValueError("There must be one probability per scenario.")
            probabilities_host = np.asarray(probabilities)
            if np.any(probabilities_host < 0.0) or not np.any(probabilities_host > 0.0):
                raise ValueError(
                    "Scenario probabilities must be non-negative with positive mass."
                )
            tolerance = 128.0 * np.finfo(probabilities_host.dtype).eps
            if abs(float(np.sum(probabilities_host)) - 1.0) > tolerance:
                raise ValueError("Scenario probabilities must sum to one.")
        if not isinstance(evidence, FinanceEvidenceBinding):
            raise TypeError("evidence must be a FinanceEvidenceBinding.")
        if not isinstance(law, PhysicalLaw):
            raise TypeError("Portfolio forecasts require a PhysicalLaw.")
        as_of = int(as_of_time_ns)
        available = int(available_time_ns)
        if available < as_of:
            raise ValueError("available_time_ns cannot precede as_of_time_ns.")
        self.expected_returns = expected
        self.covariance = jnp.asarray(symmetric, dtype=expected.dtype)
        self.scenario_returns, self.scenario_probabilities = scenarios, probabilities
        self.evidence = evidence
        self.law, self.asset_ids = law, assets
        self.as_of_time_ns, self.available_time_ns = as_of, available

    @property
    def law_id(self) -> str:
        return self.law.law_id

    @property
    def asset_count(self) -> int:
        return len(self.asset_ids)

    @property
    def scenario_count(self) -> int:
        return 0 if self.scenario_returns is None else int(self.scenario_returns.shape[0])


class PortfolioScaling(StrictModule):
    """Explicit diagonal decision scaling and scalar canonical row/objective scaling."""

    weight_scale: Array
    objective_scale: float = eqx.field(static=True)
    constraint_scale: float = eqx.field(static=True)

    def __init__(
        self,
        weight_scale: ArrayLike,
        /,
        *,
        objective_scale: float = 1.0,
        constraint_scale: float = 1.0,
    ):
        scale = _real_array(weight_scale, "weight_scale", ndim=1)
        if bool(np.any(np.asarray(scale) <= 0.0)):
            raise ValueError("weight_scale must be strictly positive.")
        objective = float(objective_scale)
        constraint = float(constraint_scale)
        if not isfinite(objective) or objective <= 0.0:
            raise ValueError("objective_scale must be finite and positive.")
        if not isfinite(constraint) or constraint <= 0.0:
            raise ValueError("constraint_scale must be finite and positive.")
        self.weight_scale = scale
        self.objective_scale = objective
        self.constraint_scale = constraint


class PortfolioProblem(StrictModule):
    """One typed portfolio definition before deterministic canonical compilation."""

    forecast: ForecastLaw
    objective: PortfolioObjective
    constraints: PortfolioConstraints
    scaling: PortfolioScaling
    current_weights: Array | None
    problem_id: str = eqx.field(static=True)
    decision_time_ns: int = eqx.field(static=True)

    def __init__(
        self,
        forecast: ForecastLaw,
        objective: PortfolioObjective,
        constraints: PortfolioConstraints,
        /,
        *,
        problem_id: str,
        decision_time_ns: int,
        scaling: PortfolioScaling | None = None,
        current_weights: ArrayLike | None = None,
    ):
        if not isinstance(forecast, ForecastLaw):
            raise TypeError("forecast must be a ForecastLaw.")
        objective_types = (
            BlackLittermanObjective,
            DrawdownRiskObjective,
            FiniteScenarioKellyObjective,
            KLDivergenceRobustObjective,
            MeanVarianceObjective,
            SpectralRiskObjective,
            TrackingErrorObjective,
            CVaRObjective,
            EVaRObjective,
        )
        if not isinstance(objective, objective_types):
            raise TypeError("objective is not a supported portfolio objective.")
        if not isinstance(constraints, PortfolioConstraints):
            raise TypeError("constraints must be PortfolioConstraints.")
        count = forecast.asset_count
        scale = (
            PortfolioScaling(jnp.ones((count,), dtype=forecast.expected_returns.dtype))
            if scaling is None
            else scaling
        )
        if not isinstance(scale, PortfolioScaling) or scale.weight_scale.shape != (
            count,
        ):
            raise ValueError(f"scaling.weight_scale must have shape ({count},).")
        current = (
            None
            if current_weights is None
            else _real_array(current_weights, "current_weights", ndim=1).astype(
                forecast.expected_returns.dtype
            )
        )
        if current is not None and current.shape != (count,):
            raise ValueError(f"current_weights must have shape ({count},).")
        if (
            constraints.turnover_limit is not None or constraints.fixed_fees is not None
        ) and current is None:
            raise ValueError(
                "current_weights are required for turnover or fixed-fee constraints."
            )
        arrays = (
            (constraints.lower_weights, "lower_weights"),
            (constraints.upper_weights, "upper_weights"),
            (constraints.lot_sizes, "lot_sizes"),
            (constraints.fixed_fees, "fixed_fees"),
        )
        for value, name in arrays:
            if value is not None and value.shape != (count,):
                raise ValueError(f"{name} must have shape ({count},).")
        if (
            constraints.linear_matrix is not None
            and constraints.linear_matrix.shape[1] != count
        ):
            raise ValueError("linear_matrix must have one column per asset.")
        if any(item.nominal.shape != (count,) for item in constraints.robust):
            raise ValueError("Every robust nominal vector must have one entry per asset.")
        if (
            constraints.maximum_cardinality is not None
            and constraints.maximum_cardinality > count
        ):
            raise ValueError("maximum_cardinality cannot exceed the asset count.")
        tree = constraints.scenario_tree
        if tree is not None:
            returns = forecast.scenario_returns
            if returns is None or returns.ndim != 3:
                raise ValueError("A scenario tree requires rank-3 scenario_returns.")
            if returns.shape[:2] != (tree.scenario_count, tree.stage_count):
                raise ValueError("Scenario returns and tree history shapes disagree.")
        elif (
            forecast.scenario_returns is not None and forecast.scenario_returns.ndim == 3
        ):
            raise ValueError("Rank-3 scenario returns require an explicit scenario_tree.")
        scenario_objectives = (
            FiniteScenarioKellyObjective,
            CVaRObjective,
            EVaRObjective,
            KLDivergenceRobustObjective,
            SpectralRiskObjective,
            DrawdownRiskObjective,
        )
        if (
            isinstance(objective, scenario_objectives)
            and forecast.scenario_returns is None
        ):
            raise ValueError("The selected objective requires finite scenarios.")
        if (
            isinstance(objective, DrawdownRiskObjective)
            and forecast.scenario_returns.ndim != 3
        ):
            raise ValueError("Drawdown risk requires scenario paths with a stage axis.")
        if isinstance(
            objective, BlackLittermanObjective
        ) and objective.equilibrium_returns.shape != (count,):
            raise ValueError("Black--Litterman moments must match the asset universe.")
        identifier = str(problem_id)
        decision = int(decision_time_ns)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        if forecast.available_time_ns > decision:
            raise ValueError(
                "Forecast information was not available at decision_time_ns."
            )
        self.forecast, self.objective, self.constraints = forecast, objective, constraints
        self.scaling, self.current_weights = scale, current
        self.problem_id, self.decision_time_ns = identifier, decision


__all__ = ["ForecastLaw", "PortfolioProblem", "PortfolioScaling"]
