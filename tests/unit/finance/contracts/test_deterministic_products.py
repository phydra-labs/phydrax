#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.finance.contracts._fixed_income import ResolvedFixedRateBond
from phydrax.finance.contracts._rates import (
    FRASettlement,
    FuturesQuoteConvention,
    InflationLegStyle,
    PayReceive,
    ResolvedCrossCurrencySwap,
    ResolvedFixedLeg,
    ResolvedFloatingLeg,
    ResolvedForwardRateAgreement,
    ResolvedIborSwap,
    ResolvedInflationLeg,
    ResolvedInterestRateFuture,
    ResolvedRateSchedule,
)
from phydrax.finance.core import Currency, DayCount, FinanceDate, FXPair, ResolvedSchedule
from phydrax.finance.curves._core import (
    CurveDefinition,
    CurveGrid,
    CurveSet,
    InterpolationPolicy,
    PreparedCurve,
)


USD = Currency("USD", 2)
EUR = Currency("EUR", 2)
VALUATION_DATE = FinanceDate.from_iso("2026-09-08")
POLICY = InterpolationPolicy(
    "linear", left_extrapolation="forbid", right_extrapolation="flat_forward"
)


def _resolved_schedule(periods=2):
    start = VALUATION_DATE.ordinal
    starts = jnp.asarray(
        [start + 365 * index for index in range(periods)], dtype=jnp.int32
    )
    ends = jnp.asarray(
        [start + 365 * (index + 1) for index in range(periods)], dtype=jnp.int32
    )
    schedule = ResolvedSchedule(
        ends,
        ends,
        starts,
        ends,
        ends,
        jnp.ones((periods,)),
        jnp.ones((periods,), dtype=bool),
        "weekend",
        "calendar-snapshot",
        f"annual-{periods}",
        DayCount.ACT_365F,
    )
    return ResolvedRateSchedule(
        schedule,
        jnp.arange(periods, dtype=float),
        jnp.arange(1, periods + 1, dtype=float),
        jnp.arange(1, periods + 1, dtype=float),
        valuation_date=VALUATION_DATE,
        curve_time_day_count=DayCount.ACT_365F,
    )


def _curve(curve_id, currency, nodes, *, representation="log_discount", role="discount"):
    nodes = jnp.asarray(nodes)
    definition = CurveDefinition(
        curve_id=curve_id,
        role=role,
        valuation_date=VALUATION_DATE,
        currency=currency,
        representation=representation,
        grid=CurveGrid(jnp.arange(nodes.shape[0], dtype=float)),
        interpolation=POLICY,
    )
    return PreparedCurve(definition, nodes)


def _zero_curves():
    return CurveSet(
        (
            _curve("discount", USD, jnp.zeros((4,))),
            _curve("projection", USD, jnp.zeros((4,))),
        )
    )


def test_fixed_bond_clean_dirty_and_accrued_prices_are_distinct_and_reconcile():
    fixed_leg = ResolvedFixedLeg(
        contract_id="bond-leg",
        schedule=_resolved_schedule(),
        currency=USD,
        discount_curve_id="discount",
        notional=100.0,
        fixed_rate=0.04,
        pay_receive=PayReceive.RECEIVE,
        exchange_initial=False,
        exchange_final=True,
    )
    bond = ResolvedFixedRateBond("bond", fixed_leg)
    curves = _zero_curves()

    dirty = bond.dirty_price(curves, settlement_time=0.5)
    accrued = bond.accrued_price(0.5)
    clean = bond.clean_price(curves, settlement_time=0.5)

    np.testing.assert_allclose(accrued, 2.0, atol=1e-6)
    np.testing.assert_allclose(dirty, 108.0, atol=1e-6)
    np.testing.assert_allclose(clean + accrued, dirty, atol=1e-6)


def test_fra_start_discounting_and_futures_price_quote_conventions_are_explicit():
    schedule = _resolved_schedule(periods=1)
    simple_rate = 0.05
    projection = _curve(
        "projection",
        USD,
        -jnp.log1p(simple_rate) * jnp.arange(4, dtype=float),
        role="projection",
    )
    curves = CurveSet((_curve("discount", USD, jnp.zeros((4,))), projection))
    end_settled = ResolvedForwardRateAgreement(
        contract_id="fra-end",
        schedule=schedule,
        currency=USD,
        discount_curve_id="discount",
        projection_curve_id="projection",
        notional=1_000_000.0,
        fixed_rate=0.04,
        pay_receive=PayReceive.RECEIVE,
        settlement=FRASettlement.PERIOD_END,
        fixing_known=False,
    )
    start_settled = ResolvedForwardRateAgreement(
        contract_id="fra-start",
        schedule=schedule,
        currency=USD,
        discount_curve_id="discount",
        projection_curve_id="projection",
        notional=1_000_000.0,
        fixed_rate=0.04,
        pay_receive=PayReceive.RECEIVE,
        settlement=FRASettlement.PERIOD_START,
        fixing_known=False,
    )
    future = ResolvedInterestRateFuture(
        contract_id="future",
        schedule=schedule,
        currency=USD,
        projection_curve_id="projection",
        contract_quote=94.0,
        quote_value_multiplier=1_000.0,
        convexity_adjustment=0.0,
        pay_receive=PayReceive.RECEIVE,
        quote_convention=FuturesQuoteConvention.PRICE_100_MINUS_RATE,
        fixing_known=False,
    )

    end_amount = end_settled.cashflow_replay(curves).amounts[0]
    start_amount = start_settled.cashflow_replay(curves).amounts[0]
    np.testing.assert_allclose(end_amount, 10_000.0, rtol=2e-5)
    np.testing.assert_allclose(start_amount, end_amount / 1.05, rtol=2e-5)
    np.testing.assert_allclose(future.model_quote(curves), 95.0, rtol=2e-5)
    np.testing.assert_allclose(future.present_value(curves), 1_000.0, rtol=2e-5)


def test_payer_receiver_swap_parity_is_exact_under_identical_curves():
    schedule = _resolved_schedule()
    simple_rate = 0.05
    projection = _curve(
        "projection",
        USD,
        -jnp.log1p(simple_rate) * jnp.arange(4, dtype=float),
        role="projection",
    )
    curves = CurveSet((_curve("discount", USD, jnp.zeros((4,))), projection))

    def swap(contract_id, fixed_direction, floating_direction):
        fixed = ResolvedFixedLeg(
            contract_id=f"{contract_id}-fixed",
            schedule=schedule,
            currency=USD,
            discount_curve_id="discount",
            notional=100.0,
            fixed_rate=simple_rate,
            pay_receive=fixed_direction,
            exchange_initial=False,
            exchange_final=False,
        )
        floating = ResolvedFloatingLeg(
            contract_id=f"{contract_id}-floating",
            schedule=schedule,
            currency=USD,
            discount_curve_id="discount",
            projection_curve_id="projection",
            notional=100.0,
            spread=0.0,
            gearing=1.0,
            pay_receive=floating_direction,
            fixing_values=jnp.zeros((2,)),
            fixing_mask=jnp.zeros((2,), dtype=bool),
            exchange_initial=False,
            exchange_final=False,
        )
        return ResolvedIborSwap(contract_id, fixed, floating)

    payer = swap("payer", PayReceive.PAY, PayReceive.RECEIVE)
    receiver = swap("receiver", PayReceive.RECEIVE, PayReceive.PAY)
    np.testing.assert_allclose(
        payer.present_value(curves) + receiver.present_value(curves), 0.0, atol=1e-6
    )


def test_cross_currency_swap_retains_both_notional_exchange_streams():
    schedule = _resolved_schedule()
    usd_discount = _curve("usd-discount", USD, jnp.zeros((4,)))
    usd_projection = _curve("usd-projection", USD, jnp.zeros((4,)), role="projection")
    eur_discount = _curve("eur-discount", EUR, jnp.zeros((4,)))
    eur_projection = _curve("eur-projection", EUR, jnp.zeros((4,)), role="projection")
    curves = CurveSet((usd_discount, usd_projection, eur_discount, eur_projection))
    base_leg = ResolvedFloatingLeg(
        contract_id="xccy-eur",
        schedule=schedule,
        currency=EUR,
        discount_curve_id="eur-discount",
        projection_curve_id="eur-projection",
        notional=100.0,
        spread=0.0,
        gearing=1.0,
        pay_receive=PayReceive.RECEIVE,
        fixing_values=jnp.zeros((2,)),
        fixing_mask=jnp.zeros((2,), dtype=bool),
        exchange_initial=True,
        exchange_final=True,
    )
    quote_leg = ResolvedFloatingLeg(
        contract_id="xccy-usd",
        schedule=schedule,
        currency=USD,
        discount_curve_id="usd-discount",
        projection_curve_id="usd-projection",
        notional=110.0,
        spread=0.0,
        gearing=1.0,
        pay_receive=PayReceive.PAY,
        fixing_values=jnp.zeros((2,)),
        fixing_mask=jnp.zeros((2,), dtype=bool),
        exchange_initial=True,
        exchange_final=True,
    )
    swap = ResolvedCrossCurrencySwap("xccy", base_leg, quote_leg, FXPair(EUR, USD))

    replay = swap.cashflow_replay(curves)
    assert int(jnp.sum(replay.notional_exchange_mask)) == 4
    assert set(replay.slot_currency_codes) == {"EUR", "USD"}
    np.testing.assert_allclose(
        swap.present_value(curves, reporting_currency=USD, spot_quote_per_base=1.1),
        0.0,
        atol=1e-6,
    )


def test_inflation_leg_distinguishes_known_and_projected_lagged_fixings():
    schedule = _resolved_schedule()
    discount = _curve("discount", USD, jnp.zeros((4,)))
    index = _curve(
        "cpi",
        USD,
        jnp.array([100.0, 110.0, 121.0, 133.1]),
        representation="zero_rate",
        role="inflation-index",
    )
    curves = CurveSet((discount, index))
    leg = ResolvedInflationLeg(
        contract_id="inflation",
        schedule=schedule,
        currency=USD,
        discount_curve_id="discount",
        index_curve_id="cpi",
        notional=100.0,
        spread=0.0,
        pay_receive=PayReceive.RECEIVE,
        style=InflationLegStyle.YEAR_ON_YEAR,
        observation_start_times=jnp.array([0.0, 1.0]),
        observation_end_times=jnp.array([1.0, 2.0]),
        start_index_values=jnp.array([100.0, 0.0]),
        end_index_values=jnp.array([110.0, 0.0]),
        start_fixing_mask=jnp.array([True, False]),
        end_fixing_mask=jnp.array([True, False]),
    )

    replay = leg.cashflow_replay(curves)
    np.testing.assert_array_equal(replay.known_mask, jnp.array([True, False]))
    np.testing.assert_array_equal(replay.projected_mask, jnp.array([False, True]))
    np.testing.assert_allclose(replay.amounts, jnp.array([10.0, 10.0]), rtol=2e-6)
    assert leg.known_cashflows.active_count == 1
