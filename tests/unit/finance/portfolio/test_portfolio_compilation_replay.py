import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.finance.core import (
    AssetReference,
    Currency,
    CurrencyAmount,
    FinanceEvidenceBinding,
    FinancialIdentifier,
    PhysicalLaw,
)
from phydrax.finance.portfolio._compile import (
    compile_portfolio_problem,
    decode_portfolio_decision,
    portfolio_result_from_native,
    refresh_portfolio_compilation,
)
from phydrax.finance.portfolio._constraints import PortfolioConstraints, ScenarioTree
from phydrax.finance.portfolio._ledger import CashBalance, LedgerTrade, PortfolioLedger
from phydrax.finance.portfolio._objectives import MeanVarianceObjective
from phydrax.finance.portfolio._problem import ForecastLaw, PortfolioProblem
from phydrax.finance.portfolio._replay import (
    PortfolioReplayMarket,
    replay_self_financing,
    ReplayCostInputs,
)
from phydrax.optim import MixedIntegerProgram, QuadraticProgram, solve_quadratic_program


def _evidence():
    return FinanceEvidenceBinding(("data",), ("model",), ("numerics",), ("use",))


def _forecast(*, expected=(0.0, 0.0), covariance=((1.0, 0.0), (0.0, 4.0))):
    return ForecastLaw(
        ("asset:a", "asset:b"),
        jnp.asarray(expected),
        jnp.asarray(covariance),
        law=PhysicalLaw("forecast", "estimated", "assets", "decision"),
        as_of_time_ns=1,
        available_time_ns=2,
        evidence=_evidence(),
    )


def test_two_asset_minimum_variance_has_analytic_allocation():
    problem = PortfolioProblem(
        _forecast(),
        MeanVarianceObjective(1.0, return_weight=0.0),
        PortfolioConstraints(lower_weights=jnp.zeros(2), upper_weights=jnp.ones(2)),
        problem_id="two-asset",
        decision_time_ns=2,
    )
    compiled = compile_portfolio_problem(problem)
    assert isinstance(compiled.program, QuadraticProgram)

    native = solve_quadratic_program(compiled.program)
    result = portfolio_result_from_native(compiled, native, feasibility_tolerance=2e-5)

    np.testing.assert_allclose(
        result.decision.weights, jnp.asarray((0.8, 0.2)), atol=2e-4
    )
    assert bool(result.certificate.certified)
    assert result.forecast_law_id == "forecast"


def test_singular_covariance_is_accepted_without_artificial_inverse():
    problem = PortfolioProblem(
        _forecast(covariance=((1.0, 1.0), (1.0, 1.0))),
        MeanVarianceObjective(1.0),
        PortfolioConstraints(lower_weights=jnp.zeros(2), upper_weights=jnp.ones(2)),
        problem_id="singular",
        decision_time_ns=2,
    )
    compiled = compile_portfolio_problem(problem)
    assert isinstance(compiled.program, QuadraticProgram)


def test_mip_decode_returns_lots_and_activity_and_rejects_corruption():
    problem = PortfolioProblem(
        _forecast(expected=(0.1, 0.0), covariance=((0.0, 0.0), (0.0, 0.0))),
        MeanVarianceObjective(0.0),
        PortfolioConstraints(
            lower_weights=jnp.zeros(2),
            upper_weights=jnp.ones(2),
            lot_sizes=jnp.asarray((0.5, 0.5)),
            maximum_cardinality=1,
            fixed_fees=jnp.asarray((0.01, 0.01)),
        ),
        problem_id="mixed",
        decision_time_ns=2,
        current_weights=jnp.asarray((0.0, 1.0)),
    )
    compiled = compile_portfolio_problem(problem)
    assert isinstance(compiled.program, MixedIntegerProgram)
    feasible = jnp.asarray((1.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 1.0))
    decision = decode_portfolio_decision(
        compiled, feasible, structure_id=compiled.plan.structure_id
    )
    np.testing.assert_array_equal(decision.lot_counts, jnp.asarray((2, 0)))
    np.testing.assert_array_equal(decision.active, jnp.asarray((1, 0)))
    np.testing.assert_array_equal(decision.fees_activated, jnp.asarray((1, 1)))

    corrupted = feasible.at[6].set(0.0)
    with pytest.raises(ValueError, match="violates"):
        decode_portfolio_decision(
            compiled, corrupted, structure_id=compiled.plan.structure_id
        )
    with pytest.raises(ValueError, match="structure"):
        decode_portfolio_decision(compiled, feasible, structure_id="wrong")


def test_scenario_tree_enforces_information_causality():
    tree = ScenarioTree(jnp.asarray(((0, 1), (0, 2)), dtype=jnp.int32))
    forecast = ForecastLaw(
        ("asset:a", "asset:b"),
        jnp.zeros(2),
        jnp.eye(2),
        law=PhysicalLaw("tree-law", "estimated", "assets", "decision"),
        as_of_time_ns=1,
        available_time_ns=2,
        evidence=_evidence(),
        scenario_returns=jnp.asarray(
            (((0.1, 0.0), (0.2, 0.0)), ((0.0, 0.1), (0.0, 0.2)))
        ),
        scenario_probabilities=jnp.asarray((0.5, 0.5)),
    )
    problem = PortfolioProblem(
        forecast,
        MeanVarianceObjective(0.0),
        PortfolioConstraints(scenario_tree=tree),
        problem_id="tree",
        decision_time_ns=2,
    )
    compiled = compile_portfolio_problem(problem)
    causal = jnp.asarray((0.5, 0.5, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0))
    decoded = decode_portfolio_decision(
        compiled, causal, structure_id=compiled.plan.structure_id
    )
    assert decoded.weights.shape == (2, 2, 2)

    leaking = causal.at[4:6].set(jnp.asarray((0.25, 0.75)))
    with pytest.raises(ValueError, match="violates"):
        decode_portfolio_decision(
            compiled, leaking, structure_id=compiled.plan.structure_id
        )


def test_numeric_refresh_refuses_structural_change():
    constraints = PortfolioConstraints(
        lower_weights=jnp.zeros(2), upper_weights=jnp.ones(2)
    )
    problem = PortfolioProblem(
        _forecast(),
        MeanVarianceObjective(1.0),
        constraints,
        problem_id="refresh",
        decision_time_ns=2,
    )
    compiled = compile_portfolio_problem(problem)
    numeric = PortfolioProblem(
        _forecast(expected=(0.2, 0.1), covariance=((2.0, 0.0), (0.0, 3.0))),
        MeanVarianceObjective(2.0),
        constraints,
        problem_id="refresh",
        decision_time_ns=2,
    )
    refreshed = refresh_portfolio_compilation(compiled, numeric)
    assert int(refreshed.numeric_version) == 1
    assert refreshed.plan.structure_id == compiled.plan.structure_id

    structural = PortfolioProblem(
        _forecast(),
        MeanVarianceObjective(1.0),
        PortfolioConstraints(upper_weights=jnp.ones(2)),
        problem_id="refresh",
        decision_time_ns=2,
    )
    with pytest.raises(ValueError, match="structure"):
        refresh_portfolio_compilation(compiled, structural)


def test_multicurrency_settlement_replay_is_self_financing():
    usd, eur = Currency("USD", 2), Currency("EUR", 2)
    us = AssetReference(FinancialIdentifier("asset", "us"), "equity", usd, "US")
    eu = AssetReference(FinancialIdentifier("asset", "eu"), "equity", eur, "EU")
    zero_usd, zero_eur = CurrencyAmount(usd, 0), CurrencyAmount(eur, 0)
    trades = (
        LedgerTrade(
            "a-buy-us",
            "main",
            us,
            1.0,
            CurrencyAmount(usd, -10_000),
            zero_usd,
            zero_usd,
            execution_index=0,
            settlement_index=1,
        ),
        LedgerTrade(
            "b-buy-eu",
            "main",
            eu,
            2.0,
            CurrencyAmount(eur, -20_000),
            zero_eur,
            zero_eur,
            execution_index=0,
            settlement_index=2,
        ),
    )
    ledger = PortfolioLedger(
        "ledger",
        usd,
        initial_cash=(
            CashBalance("main", CurrencyAmount(usd, 1_000_000)),
            CashBalance("main", CurrencyAmount(eur, 1_000_000)),
        ),
        trades=trades,
    )
    forecast = ForecastLaw(
        (us.asset_id, eu.asset_id),
        jnp.zeros(2),
        jnp.eye(2),
        law=PhysicalLaw("replay-law", "estimated", "assets", "decision"),
        as_of_time_ns=1,
        available_time_ns=2,
        evidence=_evidence(),
    )
    problem = PortfolioProblem(
        forecast,
        MeanVarianceObjective(1.0),
        PortfolioConstraints(),
        problem_id="replay",
        decision_time_ns=2,
    )
    market = PortfolioReplayMarket(
        (us, eu),
        (usd, eur),
        jnp.asarray(((100.0, 100.0), (101.0, 100.0), (102.0, 101.0))),
        jnp.asarray(((1.0, 1.2), (1.0, 1.25), (1.0, 1.3))),
        jnp.zeros((2, 2)),
        jnp.zeros((2, 2)),
        jnp.zeros((2, 2)),
        jnp.zeros((3, 2)),
        base_currency=usd,
    )
    realized = replay_self_financing(
        ledger,
        market,
        problem,
        ReplayCostInputs(jnp.zeros(2), jnp.zeros(2)),
    )
    assert float(realized.unsettled_payables[0, 0]) == pytest.approx(100.0)
    assert float(realized.unsettled_payables[0, 1]) == pytest.approx(200.0)
    assert float(realized.unsettled_payables[1, 0]) == pytest.approx(0.0)
    assert float(realized.unsettled_payables[1, 1]) == pytest.approx(200.0)
    assert float(realized.unsettled_payables[2, 1]) == pytest.approx(0.0)
    np.testing.assert_allclose(realized.self_financing_residual, 0.0, atol=1e-10)
    assert bool(realized.valid)
