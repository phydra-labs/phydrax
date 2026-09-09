#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..core import FinancialScenarioSet


class ScenarioEvaluationAdapter(StrictModule):
    """Explicit affine factor-to-PnL adapter with no opaque payoff callable."""

    factor_loading: Array
    intercept: Array
    aggregation: Literal["node", "terminal", "sum"] = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)

    def __init__(
        self,
        factor_loading: ArrayLike,
        /,
        *,
        intercept: ArrayLike = 0.0,
        aggregation: Literal["node", "terminal", "sum"] = "terminal",
        adapter_id: str = "affine-scenario-evaluation",
    ):
        loading = jnp.asarray(factor_loading)
        if loading.ndim != 1 or loading.shape[0] == 0:
            raise ValueError("factor_loading must be a non-empty vector.")
        if jnp.issubdtype(loading.dtype, jnp.complexfloating):
            raise TypeError("factor_loading must be real-valued.")
        loading = loading.astype(jnp.result_type(loading.dtype, jnp.float32))
        intercept_ = jnp.asarray(intercept, dtype=loading.dtype)
        if intercept_.shape != ():
            raise ValueError("intercept must be scalar.")
        if not np.all(np.isfinite(np.asarray(loading))) or not np.isfinite(
            float(np.asarray(intercept_))
        ):
            raise ValueError("Scenario adapter coefficients must be finite.")
        if aggregation not in ("node", "terminal", "sum"):
            raise ValueError("aggregation must be 'node', 'terminal', or 'sum'.")
        identifier = str(adapter_id)
        if not identifier:
            raise ValueError("adapter_id must be non-empty.")
        self.factor_loading, self.intercept = loading, intercept_
        self.aggregation, self.adapter_id = aggregation, identifier


class ScenarioEvaluation(StrictModule):
    node_pnl: Array
    aggregate_pnl: Array
    probabilities: Array
    valid: Array
    scenario_semantic_id: str = eqx.field(static=True)
    scenario_numeric_id: str = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)


class ScenarioReductionResult(StrictModule):
    scenarios: FinancialScenarioSet
    representative_indices: Array
    assignment: Array
    distortion: Array
    source_numeric_id: str = eqx.field(static=True)


class ScenarioReweightingResult(StrictModule):
    scenarios: FinancialScenarioSet
    likelihood_ratio: Array
    effective_sample_size: Array
    source_numeric_id: str = eqx.field(static=True)


def evaluate_scenarios(
    scenarios: FinancialScenarioSet,
    adapter: ScenarioEvaluationAdapter,
    /,
) -> ScenarioEvaluation:
    """Evaluate active scenario nodes and preserve validity/probability evidence."""

    if not isinstance(scenarios, FinancialScenarioSet):
        raise TypeError("scenarios must be a FinancialScenarioSet.")
    if not isinstance(adapter, ScenarioEvaluationAdapter):
        raise TypeError("adapter must be a ScenarioEvaluationAdapter.")
    if adapter.factor_loading.shape != (scenarios.factor_count,):
        raise ValueError("Adapter factor dimension does not match the scenario set.")
    node = jnp.sum(scenarios.values * adapter.factor_loading, axis=-1) + adapter.intercept
    node = jnp.where(scenarios.valid, node, 0.0)
    if adapter.aggregation == "node":
        aggregate = node
    elif adapter.aggregation == "sum":
        aggregate = jnp.sum(node, axis=1)
    else:
        counts = jnp.sum(scenarios.valid, axis=1, dtype=jnp.int32)
        aggregate = node[jnp.arange(scenarios.scenario_capacity), counts - 1]
    return ScenarioEvaluation(
        node_pnl=node,
        aggregate_pnl=aggregate,
        probabilities=scenarios.weights,
        valid=scenarios.valid,
        scenario_semantic_id=scenarios.semantic_id,
        scenario_numeric_id=scenarios.numeric_id,
        adapter_id=adapter.adapter_id,
    )


def reduce_scenarios(
    scenarios: FinancialScenarioSet,
    count: int,
    /,
    *,
    numeric_id: str,
) -> ScenarioReductionResult:
    """Deterministic weighted farthest-first reduction with nearest mass transfer."""

    if not isinstance(scenarios, FinancialScenarioSet):
        raise TypeError("scenarios must be a FinancialScenarioSet.")
    if isinstance(count, bool) or int(count) <= 0:
        raise ValueError("count must be a positive integer.")
    active = np.flatnonzero(np.asarray(scenarios.scenario_active))
    requested = int(count)
    if requested > active.size:
        raise ValueError("count cannot exceed the number of active scenarios.")
    values = np.asarray(scenarios.values)[active].reshape((active.size, -1))
    probabilities = np.asarray(scenarios.weights)[active]
    selected_local = [int(np.argmax(probabilities))]
    minimum_distance = np.sum((values - values[selected_local[0]]) ** 2, axis=1)
    while len(selected_local) < requested:
        score = minimum_distance * probabilities
        score[np.asarray(selected_local, dtype=np.int64)] = -1.0
        candidate = int(np.argmax(score))
        selected_local.append(candidate)
        distance = np.sum((values - values[candidate]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
    representatives = values[np.asarray(selected_local)]
    distances = np.sum((values[:, None, :] - representatives[None, :, :]) ** 2, axis=-1)
    assignment_local = np.argmin(distances, axis=1)
    reduced_weights = np.zeros((requested,), dtype=probabilities.dtype)
    for source, destination in enumerate(assignment_local):
        reduced_weights[int(destination)] += probabilities[source]
    selected = active[np.asarray(selected_local)]
    reduced = FinancialScenarioSet(
        jnp.asarray(np.asarray(scenarios.values)[selected]),
        jnp.asarray(reduced_weights),
        scenarios.times,
        jnp.asarray(np.asarray(scenarios.valid)[selected]),
        scenarios.law_id,
        scenarios.factor_layout_id,
        scenarios.semantic_id,
        str(numeric_id),
    )
    assignment = np.full((scenarios.scenario_capacity,), -1, dtype=np.int32)
    assignment[active] = assignment_local
    distortion = np.sum(probabilities * np.min(distances, axis=1))
    return ScenarioReductionResult(
        scenarios=reduced,
        representative_indices=jnp.asarray(selected, dtype=jnp.int32),
        assignment=jnp.asarray(assignment),
        distortion=jnp.asarray(distortion),
        source_numeric_id=scenarios.numeric_id,
    )


def reweight_scenarios(
    scenarios: FinancialScenarioSet,
    likelihood_ratio: ArrayLike,
    /,
    *,
    numeric_id: str,
) -> ScenarioReweightingResult:
    """Apply an explicit Radon--Nikodym likelihood ratio and normalize once."""

    if not isinstance(scenarios, FinancialScenarioSet):
        raise TypeError("scenarios must be a FinancialScenarioSet.")
    ratio = jnp.asarray(likelihood_ratio, dtype=scenarios.weights.dtype)
    if ratio.shape != scenarios.weights.shape:
        raise ValueError("likelihood_ratio must match the scenario-capacity axis.")
    active = np.asarray(scenarios.scenario_active)
    ratio_host = np.asarray(ratio)
    if np.any(~np.isfinite(ratio_host[active])) or np.any(ratio_host[active] < 0.0):
        raise ValueError("Active likelihood ratios must be finite and non-negative.")
    if np.any(ratio_host[~active] != 0.0):
        raise ValueError("Inactive likelihood ratios must be neutral zero padding.")
    unnormalized = np.asarray(scenarios.weights) * ratio_host
    total = float(np.sum(unnormalized))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Likelihood reweighting must retain positive finite mass.")
    weights = unnormalized / total
    reduced = FinancialScenarioSet(
        scenarios.values,
        jnp.asarray(weights),
        scenarios.times,
        scenarios.valid,
        scenarios.law_id,
        scenarios.factor_layout_id,
        scenarios.semantic_id,
        str(numeric_id),
    )
    effective = 1.0 / np.sum(weights[active] ** 2)
    return ScenarioReweightingResult(
        scenarios=reduced,
        likelihood_ratio=ratio,
        effective_sample_size=jnp.asarray(effective),
        source_numeric_id=scenarios.numeric_id,
    )


def entropy_tilt_scenarios(
    scenarios: FinancialScenarioSet,
    scores: ArrayLike,
    temperature: float,
    /,
    *,
    numeric_id: str,
) -> ScenarioReweightingResult:
    """Exponentially tilt a finite scenario law through the explicit reweighting adapter."""

    score = jnp.asarray(scores, dtype=scenarios.weights.dtype)
    if score.shape != scenarios.weights.shape:
        raise ValueError("scores must match the scenario-capacity axis.")
    temperature_ = float(temperature)
    if not np.isfinite(temperature_) or temperature_ <= 0.0:
        raise ValueError("temperature must be finite and positive.")
    active = scenarios.scenario_active
    active_scores = jnp.where(active, score, -jnp.inf)
    centered = active_scores - jnp.max(active_scores)
    ratio = jnp.where(active, jnp.exp(centered / temperature_), 0.0)
    return reweight_scenarios(scenarios, ratio, numeric_id=numeric_id)


__all__ = [
    "ScenarioEvaluation",
    "ScenarioEvaluationAdapter",
    "ScenarioReductionResult",
    "ScenarioReweightingResult",
    "entropy_tilt_scenarios",
    "evaluate_scenarios",
    "reduce_scenarios",
    "reweight_scenarios",
]
