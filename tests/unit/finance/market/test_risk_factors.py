#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

from phydrax.finance.core import FinancialTimestamp, TemporalAdmissibilityPolicy
from phydrax.finance.market._events import MarketEventStream
from phydrax.finance.market._lineage import DataLineage
from phydrax.finance.market._quotes import QuoteKey, QuoteObservation
from phydrax.finance.market._risk_factors import (
    MarketState,
    RiskFactorKey,
    RiskFactorLayout,
)
from phydrax.finance.market._snapshots import MarketDataSnapshot
from phydrax.finance.market._status import MarketStatus


_POLICY = TemporalAdmissibilityPolicy(True, True, True)
_LINEAGE = DataLineage("test-feed", "risk-factors")


def _time(event: int, available: int, vintage: str) -> FinancialTimestamp:
    return FinancialTimestamp(
        event, available - 2, available - 1, available, vintage, _POLICY
    )


def _layout() -> RiskFactorLayout:
    return RiskFactorLayout(
        (
            RiskFactorKey("equity", QuoteKey("EQ", "close")),
            RiskFactorKey("rate", QuoteKey("RATE", "zero")),
        )
    )


def test_market_state_reorders_by_factor_identity_not_array_position() -> None:
    layout = _layout()
    state = MarketState(
        layout,
        jnp.asarray([101.0, 0.03]),
        jnp.asarray([True, True]),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.asarray([10, 20]),
        jnp.asarray([11, 21]),
        decision_time_ns=30,
        observation_ids=("equity-v1", "rate-v1"),
    )
    reversed_layout = RiskFactorLayout(tuple(reversed(layout.keys)))

    reordered = state.reorder(reversed_layout)

    assert jnp.array_equal(reordered.values, jnp.asarray([0.03, 101.0]))
    assert reordered.observation_ids == ("rate-v1", "equity-v1")
    assert jnp.isclose(reordered.value("equity"), 101.0)


def test_reordering_to_a_superset_marks_only_the_missing_factor() -> None:
    source_layout = _layout()
    state = MarketState(
        source_layout,
        jnp.asarray([101.0, 0.03]),
        jnp.asarray([True, True]),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.asarray([10, 20]),
        jnp.asarray([11, 21]),
        decision_time_ns=30,
    )
    credit = RiskFactorKey("credit", QuoteKey("CREDIT", "spread"))
    target = RiskFactorLayout(source_layout.keys + (credit,))

    reordered = state.reorder(target)

    assert bool(reordered.valid_mask[0])
    assert bool(reordered.valid_mask[1])
    assert not bool(reordered.valid_mask[2])
    assert int(reordered.status[2]) == int(MarketStatus.MISSING_FACTOR)
    assert float(reordered.values[2]) == 0.0


def test_snapshot_missing_factor_is_neutral_and_fail_closed() -> None:
    layout = _layout()
    equity = QuoteObservation(
        layout.keys[0].quote_key,
        101.0,
        _time(10, 11, "eq-v1"),
        _LINEAGE,
    )
    snapshot = MarketDataSnapshot((equity,), snapshot_time=_time(100, 100, "archive"))

    state = snapshot.prepare(layout, _time(20, 20, "decision"))

    assert bool(state.valid_mask[0])
    assert not bool(state.valid_mask[1])
    assert int(state.status[1]) == int(MarketStatus.MISSING_FACTOR)
    assert float(state.values[1]) == 0.0


def test_fixed_shape_preparation_surfaces_duplicate_latest_vintages() -> None:
    factor = RiskFactorKey("fixing", QuoteKey("IDX", "fixing"))
    layout = RiskFactorLayout((factor,))
    observations = (
        QuoteObservation(factor.quote_key, 1.0, _time(10, 12, "v1"), _LINEAGE),
        QuoteObservation(factor.quote_key, 2.0, _time(10, 12, "v2"), _LINEAGE),
    )
    snapshot = MarketDataSnapshot(
        observations,
        snapshot_time=_time(100, 100, "duplicate-archive"),
    )

    state = snapshot.prepare(layout, _time(20, 20, "duplicate-decision"))

    assert not bool(state.valid_mask[0])
    assert int(state.status[0]) == int(MarketStatus.DUPLICATE)
    assert float(state.values[0]) == 0.0


def test_event_stream_capacity_overflow_invalidates_the_replay() -> None:
    layout = _layout()
    observations = (
        QuoteObservation(layout.keys[0].quote_key, 100.0, _time(10, 11, "e1"), _LINEAGE),
        QuoteObservation(layout.keys[1].quote_key, 0.02, _time(10, 12, "r1"), _LINEAGE),
        QuoteObservation(layout.keys[0].quote_key, 101.0, _time(20, 21, "e2"), _LINEAGE),
    )
    stream = MarketEventStream(observations, archive_time=_time(100, 100, "archive"))

    prepared = stream.prepare(layout, capacity=2)
    replay = prepared.replay((_time(30, 30, "decision"),))

    assert int(prepared.preparation_status) == int(MarketStatus.CAPACITY_EXCEEDED)
    assert not bool(jnp.any(replay.valid_mask))
    assert bool(jnp.all((replay.status & int(MarketStatus.CAPACITY_EXCEEDED)) != 0))


def test_replay_uses_correction_only_after_availability() -> None:
    layout = RiskFactorLayout((RiskFactorKey("fixing", QuoteKey("IDX", "fixing")),))
    observations = (
        QuoteObservation(layout.keys[0].quote_key, 1.0, _time(10, 12, "v1"), _LINEAGE),
        QuoteObservation(layout.keys[0].quote_key, 2.0, _time(10, 22, "v2"), _LINEAGE),
    )
    stream = MarketEventStream(observations, archive_time=_time(100, 100, "archive"))

    replay = stream.prepare(layout, capacity=2).replay(
        (_time(20, 20, "decision-20"), _time(30, 30, "decision-30"))
    )

    assert jnp.array_equal(replay.values[:, 0], jnp.asarray([1.0, 2.0]))
    assert bool(jnp.all(replay.valid_mask))
