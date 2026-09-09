import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.finance.core import FinancialScenarioSet, PhysicalLaw, StressLaw
from phydrax.finance.market import QuoteKey, RiskFactorKey, RiskFactorLayout
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


def _physical():
    return PhysicalLaw("physical", "historical", "factors", "daily")


def test_cvar_fractional_atom_and_spectral_mixture_are_exact():
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


def test_factor_risk_components_reconcile_total_volatility():
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


def test_kelly_bankruptcy_is_reported_instead_of_clipped():
    result = kelly_risk(
        jnp.asarray((-1.0, 0.1)),
        jnp.asarray((0.5, 0.5)),
        _physical(),
    )
    assert bool(result.bankrupt[0])
    assert float(result.bankruptcy_probability) == pytest.approx(0.5)
    assert bool(jnp.isneginf(result.expected_log_growth))


def test_scenario_evaluation_reduction_and_reweighting_preserve_meaning():
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
        jnp.ones((3, 2), dtype=bool),
        "physical",
        "factors",
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


def test_stress_law_is_distinct_and_pnl_explanation_reconciles():
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


def test_attribution_reconciles_active_return():
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


def test_walk_forward_rejects_information_leakage_and_nested_splits_are_contained():
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


def test_nested_backtest_selects_only_from_inner_holdouts():
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
