#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

from phydrax.finance.core import FinancialTimestamp, TemporalAdmissibilityPolicy
from phydrax.finance.market._events import PointInTimePanel
from phydrax.finance.market._lineage import DataLineage
from phydrax.finance.market._quotes import (
    FixingSeries,
    QuoteKey,
    QuoteObservation,
    QuoteTiePolicy,
)
from phydrax.finance.market._risk_factors import RiskFactorKey, RiskFactorLayout
from phydrax.finance.market._snapshots import MarketDataSnapshot
from phydrax.finance.market._status import MarketStatus


_POLICY = TemporalAdmissibilityPolicy(True, True, True)


def _timestamp(event: int, available: int, vintage: str) -> FinancialTimestamp:
    return FinancialTimestamp(
        event,
        available - 2,
        available - 1,
        available,
        vintage,
        _POLICY,
    )


def _lineage(dataset: str = "quotes") -> DataLineage:
    return DataLineage("exchange-feed", dataset, publisher_id="venue")


def test_future_correction_is_excluded_until_its_availability_clock() -> None:
    key = QuoteKey("SOFR", "fixing", venue="administrator")
    original = QuoteObservation(key, 0.051, _timestamp(10, 20, "v1"), _lineage())
    correction = QuoteObservation(key, 0.052, _timestamp(10, 40, "v2"), _lineage())
    series = FixingSeries(key, (correction, original))

    historical = series.at(
        _timestamp(10, 10, "query-event"), _timestamp(30, 30, "decision-30")
    )
    corrected = series.at(
        _timestamp(10, 10, "query-event-2"), _timestamp(50, 50, "decision-50")
    )

    assert bool(historical.valid)
    assert jnp.isclose(historical.value, 0.051)
    assert historical.observation_id == original.observation_id
    assert bool(corrected.valid)
    assert jnp.isclose(corrected.value, 0.052)
    assert corrected.observation_id == correction.observation_id
    assert series.vintage_count == 2


def test_duplicate_latest_vintages_require_an_explicit_tie_policy() -> None:
    key = QuoteKey("CPI", "fixing")
    first = QuoteObservation(key, 300.0, _timestamp(10, 20, "v1"), _lineage())
    second = QuoteObservation(key, 301.0, _timestamp(10, 20, "v2"), _lineage())
    series = FixingSeries(key, (first, second))

    rejected = series.at(_timestamp(10, 10, "event"), _timestamp(30, 30, "decision"))
    selected = series.at(
        _timestamp(10, 10, "event-2"),
        _timestamp(30, 30, "decision-2"),
        tie_policy=QuoteTiePolicy.LATEST_VINTAGE,
    )

    assert not bool(rejected.valid)
    assert int(rejected.status) == int(MarketStatus.DUPLICATE)
    assert rejected.duplicate_count == 2
    assert bool(selected.valid)
    assert jnp.isclose(selected.value, 301.0)


def test_lineage_round_trip_and_derivation_preserve_data_provenance() -> None:
    raw = _lineage("raw-quotes")
    recovered = DataLineage.from_record(raw.to_record())
    derived = DataLineage.derived(
        "mid-quote",
        (recovered,),
        source_id="phydrax.finance.market",
        dataset_id="mid-quotes",
    )

    assert recovered.lineage_id == raw.lineage_id
    assert derived.upstream_lineage_ids == (raw.lineage_id,)
    assert derived.transformation_ids == ("mid-quote",)
    assert "model" not in derived.to_record()


def test_point_in_time_panel_never_fills_a_missing_event_forward() -> None:
    key = QuoteKey("AAPL", "close")
    observation = QuoteObservation(key, 100.0, _timestamp(10, 11, "v1"), _lineage())
    snapshot = MarketDataSnapshot(
        (observation,), snapshot_time=_timestamp(100, 100, "archive")
    )
    layout = RiskFactorLayout((RiskFactorKey("aapl.close", key),))

    panel = PointInTimePanel.from_snapshot(
        snapshot,
        layout,
        (_timestamp(10, 10, "event-10"), _timestamp(11, 11, "event-11")),
        (_timestamp(20, 20, "decision-20"), _timestamp(20, 20, "decision-20b")),
    )

    assert bool(panel.valid_mask[0, 0])
    assert not bool(panel.valid_mask[1, 0])
    assert int(panel.status[1, 0]) == int(MarketStatus.MISSING_FACTOR)
    assert float(panel.values[1, 0]) == 0.0


def test_snapshot_reports_staleness_crossed_quotes_and_causal_exclusion() -> None:
    bid = QuoteKey("AAPL", "bid", venue="XNAS")
    ask = QuoteKey("AAPL", "ask", venue="XNAS")
    late = QuoteKey("AAPL", "last", venue="XNAS")
    observations = (
        QuoteObservation(bid, 101.0, _timestamp(10, 20, "bid-v1"), _lineage()),
        QuoteObservation(ask, 100.0, _timestamp(10, 20, "ask-v1"), _lineage()),
        QuoteObservation(late, 100.5, _timestamp(10, 80, "last-v1"), _lineage()),
    )
    snapshot = MarketDataSnapshot(
        observations, snapshot_time=_timestamp(100, 100, "archive")
    )
    layout = RiskFactorLayout(
        (
            RiskFactorKey("bid", bid),
            RiskFactorKey("ask", ask),
            RiskFactorKey("last", late),
        )
    )
    state = snapshot.prepare(layout, _timestamp(50, 50, "decision"), max_age_ns=20)

    assert int(state.status[0]) & int(MarketStatus.CROSSED_QUOTE)
    assert int(state.status[0]) & int(MarketStatus.STALE)
    assert int(state.status[1]) & int(MarketStatus.CROSSED_QUOTE)
    assert int(state.status[2]) == int(MarketStatus.CAUSAL_TIME_VIOLATION)
    assert not bool(jnp.any(state.accepted))
