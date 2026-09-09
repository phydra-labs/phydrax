import math

import jax


jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.finance.contracts._credit import (
    credit_event_cashflows,
    CreditDefaultSwapContract,
    CreditPayoff,
    DefaultableBondContract,
    DefaultEventState,
    RecoveryTerms,
)
from phydrax.finance.core import (
    Currency,
    CurrencyAmount,
    DayCount,
    FinanceDate,
    FinancialIdentifier,
    InstrumentReference,
    PhysicalLaw,
    PricingLaw,
    ResolvedSchedule,
    StressLaw,
)
from phydrax.finance.curves._core import (
    CurveDefinition,
    CurveGrid,
    CurveRepresentation,
    ExtrapolationMode,
    InterpolationMethod,
    InterpolationPolicy,
    PreparedCurve,
)
from phydrax.finance.models._credit import (
    cds_leg_values,
    cds_par_spread,
    defaultable_bond_cashflows,
    intensity_from_factors,
    IntensityCreditModel,
    ReducedFormCreditModel,
    simulate_default_events,
    state_dependent_intensity,
    structural_default_probability,
    StructuralCreditModel,
)
from phydrax.finance.models._rates import (
    cir_plus_plus_zero_coupon_bond,
    cir_zero_coupon_bond,
    CIRModel,
    CIRPlusPlusModel,
    FiniteFactorHJMModel,
    hjm_risk_neutral_drift,
    hull_white_zero_coupon_bond,
    HullWhiteModel,
    LiborMarketModel,
    lmm_drift,
    require_tenor_compatibility,
    vasicek_zero_coupon_bond,
    VasicekModel,
)
from phydrax.stochastic import PoissonClockRealization


def _pricing(layout="rate-layout", measure="Q"):
    return PricingLaw(
        "pricing-law",
        "synthetic reference law",
        layout,
        "market-filtration",
        measure,
        "cash-account",
        "uncollateralized",
    )


def _schedule(years=4):
    starts = [FinanceDate.from_ymd(2025 + index, 1, 1) for index in range(years)]
    ends = [FinanceDate.from_ymd(2026 + index, 1, 1) for index in range(years)]
    end_ordinals = np.asarray([value.ordinal for value in ends], dtype=np.int32)
    start_ordinals = np.asarray([value.ordinal for value in starts], dtype=np.int32)
    return ResolvedSchedule(
        end_ordinals,
        end_ordinals,
        start_ordinals,
        end_ordinals,
        end_ordinals,
        np.ones((years,), dtype=float),
        np.ones((years,), dtype=bool),
        "weekdays",
        "calendar-snapshot",
        "annual-schedule",
        DayCount.ACT_365F,
    )


def _curve(currency, representation, values, *, curve_id, role, method):
    grid = CurveGrid(jnp.arange(len(values), dtype=float))
    policy = InterpolationPolicy(
        method,
        left_extrapolation=ExtrapolationMode.FORBID,
        right_extrapolation=ExtrapolationMode.FORBID,
    )
    definition = CurveDefinition(
        curve_id=curve_id,
        role=role,
        valuation_date=FinanceDate.from_ymd(2025, 1, 1),
        currency=currency,
        representation=representation,
        grid=grid,
        interpolation=policy,
    )
    return PreparedCurve(definition, jnp.asarray(values, dtype=float))


def _credit_fixture(years=4, hazard=0.02, spread=0.0):
    currency = Currency("USD", 2)
    schedule = _schedule(years)
    survival = _curve(
        currency,
        CurveRepresentation.HAZARD_RATE,
        [hazard] * (years + 1),
        curve_id="issuer-survival",
        role="survival",
        method=InterpolationMethod.STEP_LEFT,
    )
    discount = _curve(
        currency,
        CurveRepresentation.LOG_DISCOUNT,
        [0.0] * (years + 1),
        curve_id="usd-discount",
        role="discount",
        method=InterpolationMethod.LINEAR,
    )
    recovery = RecoveryTerms(
        0.4,
        convention="par",
        timing="period_end",
        terms_id="recovery-40",
    )
    model = ReducedFormCreditModel(
        survival,
        recovery,
        reference_entity_id="internal:issuer",
        factor_layout_id="credit-layout",
        pricing_measure_id="Q",
        model_id="reduced-form",
        default_process_id="issuer-default-clock",
    )
    identifier = FinancialIdentifier("internal", "cds")
    entity = FinancialIdentifier("internal", "issuer")
    instrument = InstrumentReference(
        identifier,
        (entity,),
        currency,
        "running-spread",
        "synthetic CDS",
    )
    payoff = CreditPayoff("cds-payoff", entity.canonical, "credit_default_swap")
    cds = CreditDefaultSwapContract(
        instrument,
        schedule,
        CurrencyAmount(currency, 100_00),
        spread,
        jnp.arange(1, years + 1, dtype=float),
        recovery,
        payoff,
        contract_id="cds-contract",
        protection_side="buy",
        accrued_on_default=True,
    )
    return currency, model, discount, recovery, instrument, cds


def test_affine_zero_coupon_references_cover_vasicek_hull_white_cir_and_cir_plus_plus():
    law = _pricing()
    vasicek = VasicekModel(
        0.35,
        0.04,
        0.015,
        currency_id="USD",
        state_layout_id="rate-layout",
        model_id="vasicek",
    )
    maturity = 3.0
    rate = 0.03
    b = (1.0 - math.exp(-0.35 * maturity)) / 0.35
    log_a = (0.04 - 0.015**2 / (2.0 * 0.35**2)) * (b - maturity)
    log_a -= 0.015**2 * b**2 / (4.0 * 0.35)
    expected_vasicek = math.exp(log_a - b * rate)
    np.testing.assert_allclose(
        vasicek_zero_coupon_bond(vasicek, law, rate, 0.0, maturity),
        expected_vasicek,
        rtol=2e-12,
    )

    hull_white = HullWhiteModel(
        0.2,
        0.01,
        jnp.asarray([0.0, 5.0]),
        jnp.asarray([0.006, 0.006]),
        currency_id="USD",
        state_layout_id="rate-layout",
        model_id="hull-white",
    )
    initial_discount = math.exp(-0.03 * 2.0)
    np.testing.assert_allclose(
        hull_white_zero_coupon_bond(
            hull_white,
            law,
            0.03,
            0.0,
            2.0,
            initial_discount_at_time=1.0,
            initial_discount_at_maturity=initial_discount,
            initial_forward_at_time=0.03,
        ),
        initial_discount,
        rtol=2e-12,
    )

    cir = CIRModel(
        0.5,
        0.04,
        0.1,
        currency_id="USD",
        state_layout_id="rate-layout",
        model_id="cir",
    )
    gamma = math.sqrt(0.5**2 + 2.0 * 0.1**2)
    exponential = math.exp(gamma * 2.0) - 1.0
    denominator = (gamma + 0.5) * exponential + 2.0 * gamma
    cir_b = 2.0 * exponential / denominator
    cir_a = (2.0 * gamma * math.exp((0.5 + gamma) * 2.0 / 2.0) / denominator) ** (
        2.0 * 0.5 * 0.04 / 0.1**2
    )
    expected_cir = cir_a * math.exp(-cir_b * rate)
    np.testing.assert_allclose(
        cir_zero_coupon_bond(cir, law, rate, 0.0, 2.0), expected_cir, rtol=2e-12
    )
    assert bool(cir.feller_condition)

    shifted = CIRPlusPlusModel(
        cir,
        jnp.asarray([0.0, 2.0]),
        jnp.asarray([0.01, 0.01]),
        model_id="cir-plus-plus",
    )
    np.testing.assert_allclose(
        cir_plus_plus_zero_coupon_bond(shifted, law, rate, 0.0, 2.0),
        expected_cir * math.exp(-0.02),
        rtol=2e-12,
    )
    physical = PhysicalLaw("physical", "history", "rate-layout", "history-filtration")
    with pytest.raises(TypeError, match="PricingLaw"):
        vasicek_zero_coupon_bond(vasicek, physical, rate, 0.0, maturity)


def test_hjm_and_lmm_reject_factor_tenor_and_measure_incompatibility():
    with pytest.raises(ValueError, match="shape"):
        FiniteFactorHJMModel(
            jnp.asarray([0.0, 1.0, 2.0]),
            jnp.ones((3, 1)),
            jnp.eye(2),
            factor_ids=("level", "slope"),
            currency_id="USD",
            pricing_measure_id="Q-bank-account",
            state_layout_id="hjm-layout",
            model_id="hjm",
        )
    hjm = FiniteFactorHJMModel(
        jnp.asarray([0.0, 1.0, 2.0]),
        jnp.asarray([[0.01, 0.005], [0.01, 0.004], [0.01, 0.003]]),
        jnp.asarray([[1.0, 0.2], [0.2, 1.0]]),
        factor_ids=("level", "slope"),
        currency_id="USD",
        pricing_measure_id="Q-bank-account",
        state_layout_id="hjm-layout",
        model_id="hjm",
    )
    hjm_law = _pricing("hjm-layout", "Q-bank-account")
    assert hjm_risk_neutral_drift(hjm, hjm_law, 0).shape == (3,)
    with pytest.raises(ValueError, match="tenor"):
        require_tenor_compatibility(hjm, jnp.asarray([0.5, 1.0]))

    lmm = LiborMarketModel(
        jnp.asarray([0.0, 1.0, 2.0]),
        jnp.asarray([0.02, 0.02]),
        jnp.asarray([[0.2, 0.1], [0.18, 0.09]]),
        jnp.asarray([[1.0, 0.1], [0.1, 1.0]]),
        factor_ids=("level", "slope"),
        measure="terminal",
        currency_id="USD",
        pricing_measure_id="Q-terminal",
        state_layout_id="lmm-layout",
        model_id="lmm",
    )
    with pytest.raises(ValueError, match="measure"):
        lmm_drift(lmm, _pricing("lmm-layout", "Q-spot"), jnp.asarray([0.03, 0.035]))
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="Displaced"):
        lmm_drift(
            lmm,
            _pricing("lmm-layout", "Q-terminal"),
            jnp.asarray([-0.03, 0.035]),
        )


def test_constant_hazard_cds_parity_and_defaultable_bond_recovery():
    _, model, discount, recovery, instrument, cds = _credit_fixture()
    law = _pricing("credit-layout", "Q")
    times = np.arange(1.0, 5.0)
    survival = np.exp(-0.02 * times)
    previous = np.concatenate(([1.0], survival[:-1]))
    defaults = previous - survival
    analytic_spread = 0.6 * np.sum(defaults) / (np.sum(survival) + 0.5 * np.sum(defaults))
    np.testing.assert_allclose(
        cds_par_spread(cds, model, discount, law), analytic_spread, rtol=2e-12
    )
    par_cds = CreditDefaultSwapContract(
        instrument,
        cds.schedule,
        cds.notional,
        analytic_spread,
        cds.payment_times,
        recovery,
        cds.payoff,
        contract_id="par-cds",
        protection_side="buy",
        accrued_on_default=True,
    )
    legs = cds_leg_values(par_cds, model, discount, law)
    np.testing.assert_allclose(legs.premium_leg, legs.protection_leg, rtol=2e-12)

    one_year_schedule = _schedule(1)
    bond_instrument = InstrumentReference(
        FinancialIdentifier("internal", "bond"),
        (FinancialIdentifier("internal", "issuer"),),
        instrument.settlement_currency,
        "clean-price",
        "synthetic bond",
    )
    bond = DefaultableBondContract(
        bond_instrument,
        one_year_schedule,
        CurrencyAmount(instrument.settlement_currency, 100_00),
        0.0,
        jnp.asarray([1.0]),
        recovery,
        CreditPayoff("bond-payoff", "internal:issuer", "defaultable_bond"),
        contract_id="defaultable-bond",
    )
    bond_cashflows = defaultable_bond_cashflows(bond, model, discount, law)
    expected = 100.0 * math.exp(-0.02) + 40.0 * (1.0 - math.exp(-0.02))
    np.testing.assert_allclose(bond_cashflows.present_value, expected, rtol=2e-12)

    event = DefaultEventState(
        jnp.asarray([1.0]),
        jnp.asarray([True]),
        jnp.asarray([0.4]),
        jnp.asarray([True]),
        jnp.asarray([0], dtype=jnp.int32),
        reference_entity_id="internal:issuer",
        law_id="pricing-law",
        realization_id="boundary-default",
        coupling_id="boundary-default-coupling",
        recovery_terms_id="recovery-40",
    )
    event_cashflows = credit_event_cashflows(bond, event)
    np.testing.assert_allclose(event_cashflows.scheduled, [[0.0]])
    np.testing.assert_allclose(event_cashflows.default_settlement, [[40.0]])
    np.testing.assert_allclose(event_cashflows.default_settlement_times, [1.0])


def test_reduced_form_default_clock_emits_one_recovery_event_per_path():
    _, model, _, _, _, _ = _credit_fixture(years=1, hazard=100.0)
    realization = PoissonClockRealization(
        jr.key(7),
        1,
        support=(0.0, 1.0),
        max_events_per_channel=2,
        sample_shape=(8,),
        process_id="issuer-default-clock",
    )
    events = simulate_default_events(model, _pricing("credit-layout", "Q"), realization)
    assert bool(jnp.all(events.occurred))
    assert bool(jnp.all(events.valid))
    assert bool(jnp.all((events.default_times >= 0.0) & (events.default_times <= 1.0)))
    np.testing.assert_allclose(events.recoveries, 0.4)


def test_intensity_credit_requires_explicit_physical_base_instead_of_reusing_q_hazard():
    _, base, _, _, _, _ = _credit_fixture(years=1, hazard=0.02)
    model = IntensityCreditModel(
        base,
        jnp.asarray([0.5]),
        transformation="exponential",
        factor_layout_id="credit-layout",
        model_id="issuer-intensity",
    )
    pricing_intensity = intensity_from_factors(
        model,
        _pricing("credit-layout", "Q"),
        jnp.asarray([0.0, 1.0]),
        jnp.zeros((2, 1)),
    )
    np.testing.assert_allclose(pricing_intensity, 0.02)
    physical = PhysicalLaw(
        "issuer-P", "historical intensity", "credit-layout", "P-filtration"
    )
    physical_intensity = state_dependent_intensity(
        model,
        physical,
        jnp.asarray([0.03, 0.03]),
        jnp.full((2, 1), 2.0 * math.log(2.0)),
    )
    np.testing.assert_allclose(physical_intensity, 0.06)


def test_structural_credit_does_not_equate_physical_and_pricing_default_laws():
    model = StructuralCreditModel(
        100.0,
        90.0,
        0.25,
        2.0,
        reference_entity_id="internal:issuer",
        factor_layout_id="structural-layout",
        model_id="merton",
    )
    physical = PhysicalLaw(
        "issuer-P", "historical calibration", "structural-layout", "P-filtration"
    )
    pricing = PricingLaw(
        "issuer-Q",
        "market calibration",
        "structural-layout",
        "Q-filtration",
        "Q",
        "cash-account",
        "uncollateralized",
    )
    physical_pd = structural_default_probability(model, physical, 0.01)
    pricing_pd = structural_default_probability(model, pricing, 0.05)
    assert float(physical_pd) > float(pricing_pd)
    stress = StressLaw(
        "issuer-stress", "scenario", "structural-layout", "stress-filtration"
    )
    with pytest.raises(TypeError, match="PhysicalLaw or PricingLaw"):
        structural_default_probability(model, stress, 0.01)
