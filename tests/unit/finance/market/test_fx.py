#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

from phydrax.finance.core import Currency, FXPair
from phydrax.finance.market._fx import (
    fx_triangle_consistency,
    FXConversionGraph,
    FXConversionPath,
)
from phydrax.finance.market._quotes import QuoteKey
from phydrax.finance.market._risk_factors import (
    MarketState,
    RiskFactorKey,
    RiskFactorLayout,
)
from phydrax.finance.market._status import MarketStatus


USD = Currency("USD", 2)
EUR = Currency("EUR", 2)
GBP = Currency("GBP", 2)
JPY = Currency("JPY", 0)


def _factor(name: str, base: Currency, quote: Currency) -> RiskFactorKey:
    return RiskFactorKey(name, QuoteKey(name, "mid", fx_pair=FXPair(base, quote)))


def _state(factors: tuple[RiskFactorKey, ...], values: tuple[float, ...]) -> MarketState:
    layout = RiskFactorLayout(factors)
    count = len(factors)
    return MarketState(
        layout,
        jnp.asarray(values),
        jnp.ones((count,), dtype=bool),
        jnp.zeros((count,), dtype=jnp.int32),
        jnp.zeros((count,), dtype=jnp.int64),
        jnp.zeros((count,), dtype=jnp.int64),
        decision_time_ns=0,
    )


def test_fx_graph_rejects_two_equally_short_routes() -> None:
    factors = (
        _factor("usd-eur", USD, EUR),
        _factor("eur-jpy", EUR, JPY),
        _factor("usd-gbp", USD, GBP),
        _factor("gbp-jpy", GBP, JPY),
    )

    resolution = FXConversionGraph(factors).resolve(USD, JPY)

    assert not bool(resolution.valid)
    assert resolution.path is None
    assert resolution.shortest_path_count == 2
    assert int(resolution.status) == int(MarketStatus.FX_AMBIGUOUS)


def test_explicit_fx_path_applies_orientation_and_endpoint_contract() -> None:
    usd_eur = _factor("usd-eur", USD, EUR)
    eur_jpy = _factor("eur-jpy", EUR, JPY)
    market = _state((usd_eur, eur_jpy), (0.9, 160.0))
    path = FXConversionPath(USD, JPY, (usd_eur, eur_jpy), (1, 1))

    converted = path.convert(10.0, market, source_currency=USD, target_currency=JPY)
    mismatch = path.convert(10.0, market, source_currency=USD, target_currency=GBP)

    assert bool(converted.valid)
    assert jnp.isclose(converted.values, 1440.0)
    assert not bool(mismatch.valid)
    assert int(mismatch.status) & int(MarketStatus.CURRENCY_MISMATCH)
    assert float(mismatch.values) == 0.0


def test_inverse_fx_leg_divides_by_the_quote() -> None:
    usd_eur = _factor("usd-eur", USD, EUR)
    market = _state((usd_eur,), (0.8,))
    path = FXConversionPath(EUR, USD, (usd_eur,), (-1,))

    result = path.convert(8.0, market, source_currency=EUR, target_currency=USD)

    assert bool(result.valid)
    assert jnp.isclose(result.values, 10.0)


def test_fx_triangle_reports_observable_inconsistency() -> None:
    usd_eur = _factor("usd-eur", USD, EUR)
    eur_jpy = _factor("eur-jpy", EUR, JPY)
    usd_jpy = _factor("usd-jpy", USD, JPY)
    market = _state((usd_eur, eur_jpy, usd_jpy), (0.9, 160.0, 145.0))

    result = fx_triangle_consistency(
        market,
        usd_eur,
        eur_jpy,
        usd_jpy,
        relative_tolerance=1.0e-12,
    )

    assert bool(result.valid)
    assert not bool(result.consistent)
    assert jnp.isclose(result.implied_cross, 144.0)
    assert jnp.isclose(result.quoted_cross, 145.0)
