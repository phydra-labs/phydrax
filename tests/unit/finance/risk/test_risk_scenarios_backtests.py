from typing import Any

from phydrax import config as pxconfig


pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.finance.core import FinancialScenarioSet, PhysicalLaw, StressLaw
from phydrax.finance.market import QuoteKey, RiskFactorKey, RiskFactorLayout
from phydrax.finance.risk import _scenarios as scenario_operations
from phydrax.finance.risk._attribution import (
    brinson_attribution,
    explain_pnl,
    stress_test,
)
from phydrax.finance.risk._backtest import (
    BacktestDecisions,
    evaluate_nested_backtest,
    evaluate_walk_forward,
    nested_backtest_splits,
    NestedBacktestDecisions,
    NestedBacktestPlan,
    WalkForwardPlan,
)
from phydrax.finance.risk._market import FactorRiskModel, market_factor_risk
from phydrax.finance.risk._measures import cvar_atoms, kelly_risk, spectral_risk
from phydrax.finance.risk._scenarios import (
    evaluate_scenarios,
    reduce_scenarios,
    reweight_scenarios,
    ScenarioEvaluationAdapter,
)


def _physical() -> Any:
    return PhysicalLaw("physical", "historical", "factors", "daily")


def test_risk_scenarios_backtests_scenario_1() -> None:
    losses = jnp.asarray((0.0, 1.0, 2.0))
    probabilities = jnp.asarray((0.5, 0.25, 0.25))
    atoms = cvar_atoms(losses, probabilities, 0.5)
    np.testing.assert_allclose(atoms.tail_weights, jnp.asarray((0.0, 0.5, 0.5)))
    assert float(atoms.value_at_risk) == pytest.approx(0.0)
    assert float(atoms.expected_shortfall) == pytest.approx(1.5)
    risk = spectral_risk(
        losses, probabilities, jnp.asarray((0.5, 0.75)), jnp.asarray((0.4, 0.6))
    )
    assert float(risk) == pytest.approx(0.4 * 1.5 + 0.6 * 2.0)
    factor = RiskFactorKey("market", QuoteKey("index:market", "level"))
    layout = RiskFactorLayout((factor,))
    law = PhysicalLaw("factor-law", "estimated", layout.layout_id, "daily")
    model = FactorRiskModel(
        ("asset:a", "asset:b"),
        layout,
        jnp.ones((2, 1)),
        jnp.asarray(((4.0,),)),
        jnp.asarray((1.0, 0.0)),
        law,
        model_id="one-factor",
    )
    report = market_factor_risk(jnp.asarray((0.5, 0.5)), model)
    assert float(report.total_variance) == pytest.approx(4.25)
    assert float(jnp.sum(report.component_volatility)) == pytest.approx(
        float(report.volatility)
    )
    result = kelly_risk(
        jnp.asarray((-1.0, 0.1)),
        jnp.asarray((0.5, 0.5)),
        _physical(),
    )
    assert bool(result.bankrupt[0])
    assert float(result.bankruptcy_probability) == pytest.approx(0.5)
    assert bool(jnp.isneginf(result.expected_log_growth))


def test_risk_scenarios_backtests_scenario_2() -> None:
    scenarios = FinancialScenarioSet(
        jnp.asarray(
            (
                ((0.0, 0.0), (1.0, 0.0)),
                ((0.0, 0.0), (0.0, 1.0)),
                ((0.0, 0.0), (2.0, 0.0)),
            )
        ),
        jnp.asarray((0.5, 0.25, 0.25)),
        jnp.asarray((0.0, 1.0)),
        jnp.ones((3, 2), dtype="bool"),
        _physical(),
        "returns",
        "sample-a",
    )
    evaluation = evaluate_scenarios(
        scenarios,
        ScenarioEvaluationAdapter(jnp.asarray((1.0, -1.0))),
    )
    np.testing.assert_allclose(evaluation.aggregate_pnl, jnp.asarray((1.0, -1.0, 2.0)))
    reduced = reduce_scenarios(scenarios, 2, numeric_id="sample-reduced")
    assert reduced.scenarios.semantic_id == scenarios.semantic_id
    assert float(jnp.sum(reduced.scenarios.weights)) == pytest.approx(1.0)
    reweighted = reweight_scenarios(
        scenarios,
        jnp.asarray((1.0, 2.0, 1.0)),
        numeric_id="sample-reweighted",
    )
    np.testing.assert_allclose(reweighted.scenarios.weights, jnp.asarray((0.4, 0.4, 0.2)))
    zeroed = reweight_scenarios(
        scenarios,
        jnp.asarray((1.0, 0.0, 1.0)),
        numeric_id="sample-zero-support",
    )
    np.testing.assert_allclose(zeroed.scenarios.weights, jnp.asarray((2 / 3, 1 / 3, 0.0)))
    assert zeroed.scenarios.scenario_active.tolist() == [True, True, False]
    law = StressLaw("shock", "committee", "assets", "instant")
    stressed = stress_test(
        jnp.asarray((100.0, 50.0)),
        jnp.asarray(((-0.1, -0.2), (0.05, -0.5))),
        ("selloff", "rotation"),
        law,
        loss_limit=20.0,
    )
    np.testing.assert_allclose(stressed.scenario_pnl, jnp.asarray((-20.0, -20.0)))
    with pytest.raises(TypeError, match="StressLaw"):
        stress_test(
            jnp.asarray((100.0,)),
            jnp.asarray(((-0.1,),)),
            ("bad",),
            _physical(),
        )

    explanation = explain_pnl(
        jnp.asarray((1.0, 2.0)),
        jnp.asarray((1.5, 1.0)),
        jnp.asarray((100.0, 50.0)),
        jnp.asarray((110.0, 45.0)),
        jnp.asarray((0, 1)),
        jnp.asarray((100.0, 200.0)),
        jnp.asarray((45.0, 245.0)),
        jnp.asarray((1.0, 1.2)),
        jnp.asarray((1.0, 1.3)),
        5.0,
    )
    assert float(explanation.residual) == pytest.approx(0.0, abs=1e-12)
    assert float(explanation.explained_pnl) == pytest.approx(float(explanation.total_pnl))
    result = brinson_attribution(
        jnp.asarray((0.4, 0.2, 0.4)),
        jnp.asarray((0.3, 0.3, 0.4)),
        jnp.asarray((0.10, 0.02, -0.01)),
        jnp.asarray((0, 0, 1)),
        ("growth", "defensive"),
    )
    assert float(result.residual) == pytest.approx(0.0, abs=1e-12)
    assert float(jnp.sum(result.group_total)) == pytest.approx(
        float(result.active_return)
    )


def test_risk_scenarios_backtests_scenario_3() -> None:
    returns = jnp.asarray(tuple((0.01 * index, -0.005 * index) for index in range(12)))
    plan = WalkForwardPlan(4, 2, step=2, embargo=1)
    decisions = BacktestDecisions(
        jnp.tile(jnp.asarray((0.5, 0.5)), (3, 1)),
        jnp.asarray((3, 5, 7)),
    )
    result = evaluate_walk_forward(returns, plan, decisions)
    np.testing.assert_array_equal(
        result.observation_indices, jnp.asarray((5, 6, 7, 8, 9, 10))
    )
    assert bool(result.valid)

    leaked = BacktestDecisions(decisions.weights, jnp.asarray((5, 5, 7)))
    with pytest.raises(ValueError, match="training window"):
        evaluate_walk_forward(returns, plan, leaked)

    nested = nested_backtest_splits(
        30,
        NestedBacktestPlan(
            WalkForwardPlan(12, 3, step=3),
            WalkForwardPlan(5, 2, step=2),
        ),
    )
    for outer, inner_group in zip(nested.outer, nested.inner, strict=True):
        assert all(inner.test_stop <= outer.train_stop for inner in inner_group)
        assert all(inner.train_start >= outer.train_start for inner in inner_group)
    returns = jnp.tile(jnp.asarray((0.02, -0.01)), (30, 1))
    plan = NestedBacktestPlan(
        WalkForwardPlan(12, 3, step=3),
        WalkForwardPlan(5, 2, step=2),
    )
    splits = nested_backtest_splits(30, plan)
    outer_count = len(splits.outer)
    inner_count = len(splits.inner[0])
    candidate_weights = jnp.asarray(((1.0, 0.0), (0.0, 1.0)))
    inner_weights = jnp.broadcast_to(
        candidate_weights,
        (outer_count, inner_count, 2, 2),
    )
    inner_cutoffs = jnp.asarray(
        tuple(
            tuple((inner.train_stop - 1, inner.train_stop - 1) for inner in inner_group)
            for inner_group in splits.inner
        )
    )
    outer_weights = jnp.broadcast_to(candidate_weights, (outer_count, 2, 2))
    outer_cutoffs = jnp.asarray(
        tuple((outer.train_stop - 1, outer.train_stop - 1) for outer in splits.outer)
    )
    decisions = NestedBacktestDecisions(
        inner_weights,
        inner_cutoffs,
        outer_weights,
        outer_cutoffs,
    )
    result = evaluate_nested_backtest(returns, plan, decisions)
    np.testing.assert_array_equal(result.selected_candidate, jnp.zeros(outer_count))
    np.testing.assert_allclose(result.out_of_sample_returns, 0.02)
    assert bool(result.valid)


@pytest.mark.parametrize(
    "scenario_count", [5, 64], ids=["five-scenarios", "64-scenarios"]
)
@pytest.mark.parametrize(
    "representative_count", [1, 4], ids=["one-representative", "four-representatives"]
)
def test_scenario_reduction_matches_dense_reference(
    scenario_count: int, representative_count: int
) -> None:
    index = np.arange(scenario_count, dtype=np.float64)
    paths = np.stack(
        (
            np.stack((index / scenario_count, np.sin(index)), axis=1),
            np.stack((np.cos(index), index**2 / scenario_count**2), axis=1),
        ),
        axis=1,
    )
    weights = np.arange(1, scenario_count + 1, dtype=np.float64)
    weights /= np.sum(weights)
    values = np.zeros((scenario_count + 2, 2, 2), dtype=np.float64)
    values[:scenario_count] = paths
    probabilities = np.zeros((scenario_count + 2,), dtype=np.float64)
    probabilities[:scenario_count] = weights
    valid = np.zeros((scenario_count + 2, 2), dtype=np.bool_)
    valid[:scenario_count] = True
    scenarios = FinancialScenarioSet(
        values,
        probabilities,
        np.asarray((0.0, 1.0), dtype=np.float64),
        valid,
        _physical(),
        "reference-returns",
        "reference-source",
    )
    flattened = paths.reshape((scenario_count, -1))
    pair_distance = np.sum((flattened[:, None, :] - flattened[None, :, :]) ** 2, axis=2)
    selected = [int(np.argmax(weights))]
    while len(selected) < representative_count:
        scores = weights * np.min(pair_distance[:, selected], axis=1)
        scores[np.asarray(selected, dtype=np.int64)] = -1.0
        selected.append(int(np.argmax(scores)))
    nearest = np.argmin(pair_distance[:, selected], axis=1)
    expected_weights = np.zeros((representative_count,), dtype=np.float64)
    for source, destination in enumerate(nearest):
        expected_weights[destination] += weights[source]
    expected_assignment = np.full((scenario_count + 2,), -1, dtype=np.int32)
    expected_assignment[:scenario_count] = nearest
    expected_distortion = np.sum(weights * np.min(pair_distance[:, selected], axis=1))

    result = reduce_scenarios(
        scenarios, representative_count, numeric_id="reference-reduced"
    )

    np.testing.assert_array_equal(result.representative_indices, selected)
    np.testing.assert_array_equal(result.assignment, expected_assignment)
    np.testing.assert_array_equal(result.scenarios.values, paths[selected])
    np.testing.assert_array_equal(result.scenarios.weights, expected_weights)
    np.testing.assert_array_equal(result.scenarios.valid, valid[selected])
    np.testing.assert_array_equal(result.scenarios.times, scenarios.times)
    np.testing.assert_allclose(
        result.distortion, expected_distortion, rtol=1.0e-13, atol=1.0e-14
    )
    assert result.scenarios.law == scenarios.law
    assert result.scenarios.semantic_id == scenarios.semantic_id
    assert result.scenarios.numeric_id == "reference-reduced"
    assert result.source_numeric_id == scenarios.numeric_id


def test_scenario_reduction_first_representative_wins_exact_ties() -> None:
    values = np.asarray(
        (
            ((0.0, 0.0),),
            ((2.0, 0.0),),
            ((1.0, 0.0),),
            ((0.0, 0.0),),
            ((2.0, 0.0),),
            ((0.0, 0.0),),
        ),
        dtype=np.float64,
    )
    scenarios = FinancialScenarioSet(
        values,
        np.asarray((0.4, 0.15, 0.2, 0.1, 0.15, 0.0), dtype=np.float64),
        np.asarray((0.0,), dtype=np.float64),
        np.asarray(
            ((True,), (True,), (True,), (True,), (True,), (False,)), dtype=np.bool_
        ),
        _physical(),
        "tie-returns",
        "tie-source",
    )

    result = scenario_operations.reduce_scenarios(scenarios, 2, numeric_id="tie-reduced")

    np.testing.assert_array_equal(result.representative_indices, (0, 1))
    np.testing.assert_array_equal(result.assignment, (0, 1, 0, 0, 1, -1))
    np.testing.assert_allclose(result.scenarios.weights, (0.7, 0.3), rtol=1.0e-13)
    np.testing.assert_allclose(result.distortion, 0.2, rtol=1.0e-13)
    assert result.scenarios.numeric_id == "tie-reduced"
    assert result.source_numeric_id == "tie-source"
