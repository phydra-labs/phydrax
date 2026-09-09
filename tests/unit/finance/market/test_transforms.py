#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

from phydrax.finance.core import (
    Currency,
    FinancialTimestamp,
    TemporalAdmissibilityPolicy,
)
from phydrax.finance.market._lineage import DataLineage
from phydrax.finance.market._status import MarketStatus
from phydrax.finance.market._transforms import (
    adjust_for_corporate_actions,
    build_time_bars,
    CorporateAction,
    CorporateActionKind,
    CorporateActionSeries,
    price_returns,
    realized_measure,
    RealizedMeasureKind,
    ReturnKind,
)


_POLICY = TemporalAdmissibilityPolicy(True, True, True)
USD = Currency("USD", 2)


def _time(event: int, available: int, vintage: str) -> FinancialTimestamp:
    return FinancialTimestamp(
        event, available - 2, available - 1, available, vintage, _POLICY
    )


def test_split_adjustment_removes_the_mechanical_price_jump() -> None:
    price_lineage = DataLineage("venue", "raw-prices")
    action_lineage = DataLineage("issuer", "corporate-actions")
    split = CorporateAction(
        "split-1",
        CorporateActionKind.SPLIT,
        2.0,
        _time(20, 25, "v1"),
        action_lineage,
    )
    adjusted = adjust_for_corporate_actions(
        jnp.asarray([10, 20]),
        jnp.asarray([100.0, 50.0]),
        jnp.asarray([True, True]),
        CorporateActionSeries((split,)),
        _time(30, 30, "decision"),
        price_lineage,
        price_currency=USD,
    )
    returns = price_returns(
        jnp.asarray([10, 20]),
        adjusted.adjusted_prices,
        adjusted.valid_mask,
        adjusted.lineage,
        kind=ReturnKind.SIMPLE,
    )

    assert jnp.array_equal(adjusted.adjusted_prices, jnp.asarray([50.0, 50.0]))
    assert bool(jnp.all(adjusted.valid_mask))
    assert jnp.isclose(returns.values[0], 0.0)
    assert action_lineage.lineage_id in adjusted.lineage.upstream_lineage_ids


def test_future_corporate_action_correction_does_not_change_historical_adjustment() -> (
    None
):
    action_lineage = DataLineage("issuer", "actions")
    original = CorporateAction(
        "split-1",
        CorporateActionKind.SPLIT,
        2.0,
        _time(20, 25, "v1"),
        action_lineage,
    )
    correction = CorporateAction(
        "split-1",
        CorporateActionKind.SPLIT,
        4.0,
        _time(20, 40, "v2"),
        action_lineage,
    )
    series = CorporateActionSeries((correction, original))

    historical = series.available_at(_time(30, 30, "decision-30"))
    corrected = series.available_at(_time(50, 50, "decision-50"))

    assert float(historical[0].value) == 2.0
    assert historical[0].observation_id == original.observation_id
    assert float(corrected[0].value) == 4.0
    assert corrected[0].observation_id == correction.observation_id


def test_time_bars_use_exact_half_open_windows() -> None:
    lineage = DataLineage("venue", "ticks")
    bars = build_time_bars(
        jnp.asarray([10, 19, 20, 29, 30]),
        jnp.asarray([1.0, 2.0, 10.0, 20.0, 100.0]),
        jnp.ones((5,)),
        jnp.ones((5,), dtype=bool),
        jnp.asarray([10, 20]),
        jnp.asarray([20, 30]),
        lineage,
    )

    assert jnp.array_equal(bars.open, jnp.asarray([1.0, 10.0]))
    assert jnp.array_equal(bars.close, jnp.asarray([2.0, 20.0]))
    assert jnp.array_equal(bars.high, jnp.asarray([2.0, 20.0]))
    assert jnp.array_equal(bars.observation_count, jnp.asarray([2, 2]))


def test_return_and_realized_measure_fail_closed_on_missing_endpoint() -> None:
    lineage = DataLineage("venue", "prices")
    returns = price_returns(
        jnp.asarray([10, 20, 30]),
        jnp.asarray([100.0, 101.0, 102.0]),
        jnp.asarray([True, False, True]),
        lineage,
        kind=ReturnKind.LOG,
    )
    measure = realized_measure(returns, kind=RealizedMeasureKind.VARIANCE)

    assert not bool(jnp.any(returns.valid_mask))
    assert bool(jnp.all((returns.status & int(MarketStatus.MISSING_FACTOR)) != 0))
    assert not bool(measure.valid)
    assert int(measure.status) == int(MarketStatus.MISSING_FACTOR)
    assert float(measure.value) == 0.0
