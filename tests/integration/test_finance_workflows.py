#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.dynamics import TimeGrid


def _timestamp(
    event: int, available: int, vintage: str
) -> phx.finance.FinancialTimestamp:
    policy = phx.finance.TemporalAdmissibilityPolicy(True, True, True)
    return phx.finance.FinancialTimestamp(
        event,
        available - 2,
        available - 1,
        available,
        vintage,
        policy,
    )


def test_market_curve_valuation_archive_workflow(tmp_path):
    finance = phx.finance
    usd = finance.Currency("USD", 2)
    asset = finance.AssetReference(
        finance.FinancialIdentifier("synthetic", "EQ"),
        "equity",
        usd,
        "synthetic equity",
    )
    lineage = finance.market.DataLineage("synthetic", "workflow")
    quote_key = finance.QuoteKey(asset.asset_id, "close", currency=usd)
    quote = finance.QuoteObservation(
        quote_key, 100.0, _timestamp(10, 12, "spot-v1"), lineage
    )
    reference = finance.ReferenceDataSnapshot(
        (asset,),
        (),
        (usd,),
        as_of=_timestamp(20, 20, "reference-v1"),
        lineage=lineage,
    )
    market = finance.MarketDataSnapshot(
        (quote,),
        snapshot_time=_timestamp(20, 20, "market-v1"),
        reference_data_id=reference.snapshot_id,
    )
    layout = finance.RiskFactorLayout((finance.RiskFactorKey("spot", quote_key),))
    state = market.prepare(layout, _timestamp(15, 15, "decision"))
    assert bool(state.valid_mask[0])

    curve = finance.curves.PreparedCurve(
        finance.curves.CurveDefinition(
            curve_id="usd-discount",
            role="discount",
            valuation_date=finance.FinanceDate.from_iso("2026-09-08"),
            currency=usd,
            representation="zero_rate",
            grid=finance.curves.CurveGrid(jnp.asarray([0.0, 1.0, 2.0])),
            interpolation=finance.curves.InterpolationPolicy(
                "linear",
                left_extrapolation="forbid",
                right_extrapolation="flat_forward",
            ),
        ),
        jnp.asarray([0.05, 0.05, 0.05]),
    )
    discount = curve.discount_factor(1.0)
    result = finance.valuation.evaluate_black_scholes_european(
        finance.models.BlackScholesModel(0.2),
        state.value("spot"),
        100.0,
        1.0,
        0.05,
    )
    implied = finance.valuation.invert_black_scholes_implied_volatility(
        result.value,
        state.value("spot"),
        100.0,
        1.0,
        0.05,
    )
    assert bool(result.successful)
    np.testing.assert_allclose(implied.volatility, 0.2, rtol=2e-5)
    np.testing.assert_allclose(discount, np.exp(-0.05), rtol=2e-14)

    arrays = {
        "present_value": np.asarray(result.value),
        "valid": np.asarray(result.successful),
    }
    manifest = finance.qualification.finance_result_manifest(
        "workflow-valuation",
        "workflow-run",
        arrays,
        {"present_value": "USD", "valid": "1"},
    )
    support = finance.qualification.valuation_support(
        "analytic",
        product="european-option",
        model="black-scholes",
        pricing_law="usd-risk-neutral",
    )
    archive_path = finance.qualification.archive_finance_result(
        tmp_path / "workflow.phx",
        result_manifest=manifest,
        arrays=arrays,
        replay_id="analytic-parity-and-inversion",
        support_tuples=(support,),
        law_ids=("usd-risk-neutral",),
    )
    reopened = finance.qualification.reopen_finance_result(archive_path)
    np.testing.assert_allclose(reopened.arrays["present_value"], result.value)


def test_physical_forecast_to_portfolio_decision_workflow():
    series = jnp.asarray([0.010, 0.015, 0.012, 0.020, 0.018, 0.024, 0.021, 0.027])
    fit = phx.uq.fit_arima(series, p=1)
    assert bool(fit.successful)

    evidence = phx.finance.FinanceEvidenceBinding(
        ("point-in-time-data",),
        ("arima-fit",),
        ("least-squares",),
        ("research-only",),
    )
    forecast = phx.finance.portfolio.ForecastLaw(
        ("asset:a", "asset:b"),
        jnp.asarray([fit.model.forecast(series, 1)[0], 0.005]),
        jnp.asarray([[0.01, 0.0], [0.0, 0.04]]),
        law=phx.finance.PhysicalLaw("forecast-p", "synthetic", "assets", "decision"),
        as_of_time_ns=10,
        available_time_ns=11,
        evidence=evidence,
    )
    problem = phx.finance.portfolio.PortfolioProblem(
        forecast,
        phx.finance.portfolio.MeanVarianceObjective(1.0, return_weight=0.0),
        phx.finance.portfolio.PortfolioConstraints(
            lower_weights=jnp.zeros(2),
            upper_weights=jnp.ones(2),
        ),
        problem_id="workflow-portfolio",
        decision_time_ns=11,
    )
    compiled = phx.finance.portfolio.compile_portfolio_problem(problem)
    native = phx.optim.solve_quadratic_program(compiled.program)
    result = phx.finance.portfolio.portfolio_result_from_native(
        compiled,
        native,
        feasibility_tolerance=2e-5,
    )
    assert bool(result.certificate.certified)
    np.testing.assert_allclose(
        result.decision.weights, jnp.asarray([0.8, 0.2]), atol=2e-4
    )


def test_credit_exposure_xva_workflow():
    finance = phx.finance
    profile = finance.exposure.ExposureProfile(
        jnp.asarray([0.0, 1.0, 2.0]),
        jnp.asarray([0.0, 10.0, 20.0]),
        jnp.asarray([0.0, 4.0, 8.0]),
        jnp.asarray([0.0, 10.0, 20.0]),
        jnp.asarray([0.0, 4.0, 8.0]),
        jnp.asarray([0.0, 10.0, 20.0]),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.asarray(1.0),
        jnp.asarray(False),
        0.95,
        "workflow-exposure",
        "workflow-values",
        "workflow-weights",
        None,
    )
    discount = jnp.ones(3)
    cva = finance.exposure.compute_cva(
        profile,
        jnp.asarray([0.0, 0.1, 0.1]),
        0.4,
        discount,
        counterparty_default_law_id="counterparty-q",
        recovery_terms_id="recovery",
        discount_curve_id="discount",
    )
    dva = finance.exposure.compute_dva(
        profile,
        jnp.asarray([0.0, 0.05, 0.05]),
        0.5,
        discount,
        own_default_law_id="own-q",
        recovery_terms_id="own-recovery",
        discount_curve_id="discount",
    )
    funding = finance.exposure.FundingPolicy(
        jnp.full(3, 0.02),
        jnp.full(3, 0.01),
        policy_id="funding",
        funding_curve_id="funding-curve",
    )
    margin = finance.exposure.MarginFundingPolicy(
        jnp.full(3, 0.01),
        funding_curve_id="margin-curve",
        policy_id="margin",
    )
    capital = finance.exposure.EconomicCapitalPolicy(
        jnp.asarray([0.0, 5.0, 5.0]),
        jnp.full(3, 0.1),
        policy_id="capital",
    )
    result = finance.exposure.assemble_xva(
        cva,
        dva,
        finance.exposure.compute_fva(
            profile, funding, discount, discount_curve_id="discount"
        ),
        finance.exposure.compute_mva(
            profile,
            jnp.asarray([0.0, 3.0, 3.0]),
            margin,
            discount,
            initial_margin_profile_id="im",
            discount_curve_id="discount",
        ),
        finance.exposure.compute_kva(
            profile, capital, discount, discount_curve_id="discount"
        ),
        result_id="workflow-xva",
    )
    np.testing.assert_allclose(result.decomposition_residual, 0.0, atol=1e-15)
    assert float(result.cva.adjustment) < 0.0
    assert float(result.dva.adjustment) > 0.0


def test_event_model_to_execution_control_workflow():
    model = phx.finance.execution.AlmgrenChrissModel(
        volatility=0.2,
        risk_aversion=0.1,
        temporary_impact=0.5,
        permanent_impact=0.01,
        model_id="workflow-impact",
    )
    schedule = phx.finance.execution.solve_almgren_chriss_schedule(
        model,
        TimeGrid(jnp.linspace(0.0, 1.0, 9), time_id="workflow-grid"),
        12.0,
    )
    assert float(schedule.conservation_residual) < 1e-12
    np.testing.assert_allclose(
        np.asarray(schedule.inventory)[[0, -1]], [12.0, 0.0], atol=1e-14
    )
    assert bool(jnp.all(schedule.trading_rates >= 0.0))
