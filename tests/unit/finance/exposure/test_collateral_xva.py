import jax


jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.finance.contracts._credit import DefaultEventState
from phydrax.finance.core import Currency
from phydrax.finance.exposure._collateral import (
    CloseoutConvention,
    collateral_target,
    CollateralAgreement,
    evolve_collateral,
    NettingSet,
    prepare_collateral_agreement,
)
from phydrax.finance.exposure._pathwise import (
    aggregate_exposure,
    ExposureProfile,
    ExposureSimulationPlan,
    link_wrong_way_risk,
    PathWeighting,
    PathwiseTradeValues,
    simulate_exposure,
    WrongWayRiskLink,
)
from phydrax.finance.exposure._replay import replay_xva
from phydrax.finance.exposure._xva import (
    assemble_xva,
    compute_cva,
    compute_dva,
    compute_fva,
    compute_kva,
    compute_mva,
    EconomicCapitalPolicy,
    FundingPolicy,
    MarginFundingPolicy,
)


def _agreement(currency, *, netting_set_id="netting", threshold=100.0, mta=0.0, lag=0.0):
    return CollateralAgreement(
        currency,
        netting_set_id=netting_set_id,
        threshold_receivable=threshold,
        threshold_postable=threshold,
        minimum_transfer_amount=mta,
        call_lag=lag,
        margin_period_of_risk=1.0,
        closeout_lag=0.0,
        agreement_id=f"agreement-{netting_set_id}",
    )


def _defaults(times, occurred, *, law_id, recovery=0.4, coupling_id=None):
    path_count = len(times)
    occurred_mask = jnp.asarray(occurred, dtype=bool)
    return DefaultEventState(
        jnp.asarray(times, dtype=float),
        occurred_mask,
        jnp.where(occurred_mask, recovery, 0.0),
        jnp.ones((path_count,), dtype=bool),
        jnp.zeros((path_count,), dtype=jnp.int32),
        reference_entity_id=f"entity-{law_id}",
        law_id=law_id,
        realization_id=f"realization-{law_id}",
        coupling_id=f"coupling-{law_id}" if coupling_id is None else coupling_id,
        recovery_terms_id=f"recovery-{law_id}",
    )


def _closeout():
    return CloseoutConvention(
        "risk_free", "mpor_end", "counterparty", convention_id="closeout"
    )


def test_collateral_threshold_mta_and_lag_boundaries_are_exact():
    currency = Currency("USD", 2)
    agreement = _agreement(currency, threshold=10.0, mta=5.0)
    np.testing.assert_allclose(
        collateral_target(agreement, jnp.asarray([10.0, -10.0])), 0.0
    )
    prepared = prepare_collateral_agreement(agreement, jnp.asarray([0.0, 1.0, 2.0]))
    path = evolve_collateral(prepared, jnp.asarray([[10.0, 14.0, 15.0]]))
    np.testing.assert_allclose(path.calls, [[0.0, 0.0, 5.0]])
    np.testing.assert_allclose(path.balances, [[0.0, 0.0, 5.0]])

    lagged = _agreement(currency, threshold=10.0, mta=0.0, lag=1.0)
    lagged_path = evolve_collateral(
        prepare_collateral_agreement(lagged, jnp.asarray([0.0, 1.0, 2.0])),
        jnp.asarray([[20.0, 20.0, 20.0]]),
    )
    np.testing.assert_allclose(lagged_path.calls, [[10.0, 0.0, 0.0]])
    np.testing.assert_allclose(lagged_path.balances, [[0.0, 10.0, 10.0]])


def test_grid_boundary_default_is_closed_out_after_mpor_before_positive_part():
    currency = Currency("USD", 2)
    times = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    netting = NettingSet(
        ("trade-a", "trade-b"),
        jnp.asarray([0, 1]),
        currency,
        agreement_scope_id="caller-resolved-scope",
        netting_set_id="netting",
    )
    agreement = _agreement(currency)
    plan = ExposureSimulationPlan(
        netting,
        prepare_collateral_agreement(agreement, times),
        _closeout(),
        default_dependence="independent",
        wrong_way_risk=None,
        counterparty_default_law_id="cp-Q",
        own_default_law_id="own-Q",
        pricing_law_id="rates-Q",
        discount_curve_id="usd-discount",
        plan_id="exposure-plan",
    )
    values = PathwiseTradeValues(
        times,
        jnp.asarray([[[8.0, 2.0], [10.0, 2.0], [18.0, 2.0], [28.0, 2.0]]]),
        jnp.asarray([True]),
        ("trade-a", "trade-b"),
        currency,
        pricing_law_id="rates-Q",
        factor_layout_id="rates-layout",
        realization_id="market-paths",
        coupling_id="market-coupling",
        value_state_id="trade-values",
    )
    weighting = PathWeighting(
        jnp.asarray([1.0]),
        jnp.asarray([True]),
        ("path-0",),
        iid=True,
        weighting_id="equal-weights",
    )
    exposure = simulate_exposure(
        plan,
        values,
        _defaults([1.0], [True], law_id="cp-Q"),
        _defaults([0.0], [False], law_id="own-Q"),
        jnp.ones((4,)),
        weighting,
    )
    # Netting occurs first (18 + 2 = 20), collateral is zero, and the exact
    # t=1 default closes at t=2 after MPOR. Positive part is taken only then.
    np.testing.assert_allclose(exposure.netted_values, [[10.0, 12.0, 20.0, 30.0]])
    np.testing.assert_allclose(exposure.residual_values, [[10.0, 0.0, 20.0, 0.0]])
    np.testing.assert_allclose(exposure.positive_exposure, [[10.0, 0.0, 20.0, 0.0]])


def test_invalid_path_weights_are_rejected_and_explicit_wwr_changes_exposure():
    with pytest.raises(ValueError, match="Invalid paths"):
        PathWeighting(
            jnp.asarray([0.5, 0.5]),
            jnp.asarray([True, False]),
            ("path-0", "path-1"),
            iid=True,
            weighting_id="invalid",
        )

    currency = Currency("USD", 2)
    times = jnp.asarray([0.0, 1.0])
    netting = NettingSet(
        ("trade",),
        jnp.asarray([0]),
        currency,
        agreement_scope_id="caller-resolved-scope",
        netting_set_id="netting",
    )
    prepared = prepare_collateral_agreement(_agreement(currency), times)
    values = PathwiseTradeValues(
        times,
        jnp.asarray([[[1.0], [1.0]], [[3.0], [3.0]]]),
        jnp.asarray([True, True]),
        ("trade",),
        currency,
        pricing_law_id="rates-Q",
        factor_layout_id="joint-layout",
        realization_id="shared-paths",
        coupling_id="shared-paths",
        value_state_id="marks",
    )
    weighting = PathWeighting(
        jnp.asarray([0.5, 0.5]),
        jnp.asarray([True, True]),
        ("path-0", "path-1"),
        iid=True,
        weighting_id="equal",
    )
    independent_plan = ExposureSimulationPlan(
        netting,
        prepared,
        _closeout(),
        default_dependence="independent",
        wrong_way_risk=None,
        counterparty_default_law_id="credit-Q",
        own_default_law_id="own-Q",
        pricing_law_id="rates-Q",
        discount_curve_id="usd-discount",
        plan_id="independent-plan",
    )
    cp = _defaults([0.0, 0.0], [False, False], law_id="credit-Q")
    own = _defaults([0.0, 0.0], [False, False], law_id="own-Q")
    cp_wwr = _defaults(
        [0.0, 0.0],
        [False, False],
        law_id="credit-Q",
        coupling_id="shared-paths",
    )
    independent = aggregate_exposure(
        simulate_exposure(
            independent_plan,
            values,
            cp,
            own,
            jnp.ones((2,)),
            weighting,
        )
    )
    link = WrongWayRiskLink(
        "rates-Q",
        "credit-Q",
        "shared_factor",
        ("credit-market-factor",),
        "shared-paths",
        link_id="explicit-wwr",
    )
    linked = link_wrong_way_risk(
        weighting,
        link,
        shared_factor_likelihood=jnp.asarray([0.5, 1.5]),
    )
    wwr_plan = ExposureSimulationPlan(
        netting,
        prepared,
        _closeout(),
        default_dependence="wrong_way",
        wrong_way_risk=link,
        counterparty_default_law_id="credit-Q",
        own_default_law_id="own-Q",
        pricing_law_id="rates-Q",
        discount_curve_id="usd-discount",
        plan_id="wwr-plan",
    )
    wwr = aggregate_exposure(
        simulate_exposure(
            wwr_plan,
            values,
            cp_wwr,
            own,
            jnp.ones((2,)),
            weighting,
            wrong_way_result=linked,
        )
    )
    np.testing.assert_allclose(independent.expected_positive_exposure, [2.0, 2.0])
    np.testing.assert_allclose(wwr.expected_positive_exposure, [2.5, 2.5])


def _profile():
    return ExposureProfile(
        jnp.asarray([0.0, 1.0, 2.0]),
        jnp.asarray([0.0, 10.0, 20.0]),
        jnp.asarray([0.0, 4.0, 8.0]),
        jnp.asarray([0.0, 10.0, 20.0]),
        jnp.asarray([0.0, 4.0, 8.0]),
        jnp.asarray([0.0, 10.0, 20.0]),
        jnp.asarray([0.0, 0.0, 0.0]),
        jnp.asarray([0.0, 0.0, 0.0]),
        jnp.asarray(1.0),
        jnp.asarray(False),
        0.95,
        "exposure-plan",
        "value-state",
        "weights",
        None,
    )


def test_xva_sign_decomposition_and_independent_corruption_replay():
    profile = _profile()
    discount = jnp.ones((3,))
    funding = FundingPolicy(
        jnp.full((3,), 0.02),
        jnp.full((3,), 0.01),
        funding_curve_id="funding-curve",
        policy_id="funding",
    )
    margin = MarginFundingPolicy(
        jnp.full((3,), 0.01),
        funding_curve_id="margin-funding-curve",
        policy_id="margin",
    )
    capital = EconomicCapitalPolicy(
        jnp.asarray([0.0, 5.0, 5.0]),
        jnp.full((3,), 0.1),
        policy_id="economic-capital",
    )
    cva = compute_cva(
        profile,
        jnp.asarray([0.0, 0.1, 0.1]),
        0.4,
        discount,
        counterparty_default_law_id="cp-Q",
        recovery_terms_id="cp-recovery",
        discount_curve_id="discount",
    )
    dva = compute_dva(
        profile,
        jnp.asarray([0.0, 0.05, 0.05]),
        0.5,
        discount,
        own_default_law_id="own-Q",
        recovery_terms_id="own-recovery",
        discount_curve_id="discount",
    )
    fva = compute_fva(profile, funding, discount, discount_curve_id="discount")
    mva = compute_mva(
        profile,
        jnp.asarray([0.0, 3.0, 3.0]),
        margin,
        discount,
        initial_margin_profile_id="initial-margin",
        discount_curve_id="discount",
    )
    kva = compute_kva(profile, capital, discount, discount_curve_id="discount")
    result = assemble_xva(cva, dva, fva, mva, kva, result_id="xva")
    np.testing.assert_allclose(
        [cva.adjustment, dva.adjustment, fva.adjustment, mva.adjustment, kva.adjustment],
        [-1.8, 0.3, -0.48, -0.06, -1.0],
        rtol=2e-12,
    )
    np.testing.assert_allclose(result.total_adjustment, -3.04, rtol=2e-12)
    np.testing.assert_allclose(result.decomposition_residual, 0.0, atol=1e-15)

    replay = replay_xva(
        result,
        profile,
        jnp.asarray([0.0, 0.1, 0.1]),
        0.4,
        jnp.asarray([0.0, 0.05, 0.05]),
        0.5,
        discount,
        funding,
        jnp.asarray([0.0, 3.0, 3.0]),
        margin,
        capital,
        counterparty_default_law_id="cp-Q",
        counterparty_recovery_terms_id="cp-recovery",
        own_default_law_id="own-Q",
        own_recovery_terms_id="own-recovery",
        initial_margin_profile_id="initial-margin",
        discount_curve_id="discount",
    )
    assert bool(replay.valid)

    corrupted = eqx.tree_at(
        lambda value: value.cva.adjustment,
        result,
        result.cva.adjustment + 1.0,
    )
    corrupted_replay = replay_xva(
        corrupted,
        profile,
        jnp.asarray([0.0, 0.1, 0.1]),
        0.4,
        jnp.asarray([0.0, 0.05, 0.05]),
        0.5,
        discount,
        funding,
        jnp.asarray([0.0, 3.0, 3.0]),
        margin,
        capital,
        counterparty_default_law_id="cp-Q",
        counterparty_recovery_terms_id="cp-recovery",
        own_default_law_id="own-Q",
        own_recovery_terms_id="own-recovery",
        initial_margin_profile_id="initial-margin",
        discount_curve_id="discount",
    )
    assert not bool(corrupted_replay.valid)
    assert "xva:cva:value-or-buckets" in corrupted_replay.failure_ids
