#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..core import FinancialTimestamp
from ._quotes import QuoteObservation, QuoteTiePolicy
from ._risk_factors import RiskFactorLayout
from ._snapshots import MarketDataSnapshot
from ._status import MarketStatus


class PointInTimePanel(StrictModule, NonTrainableState):
    """Exact-event bitemporal panel; missing cells are never filled forward."""

    layout: RiskFactorLayout
    event_time_ns: Array
    decision_time_ns: Array
    values: Array
    valid_mask: Array
    status: Array
    availability_time_ns: Array
    observation_ids: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    panel_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: RiskFactorLayout,
        event_time_ns: Array,
        decision_time_ns: Array,
        values: Array,
        valid_mask: Array,
        status: Array,
        availability_time_ns: Array,
        /,
        *,
        observation_ids: Sequence[Sequence[str]],
    ):
        if not isinstance(layout, RiskFactorLayout):
            raise TypeError("layout must be a RiskFactorLayout.")
        event_host = np.asarray(event_time_ns, dtype=np.int64)
        decision_host = np.asarray(decision_time_ns, dtype=np.int64)
        value_host = np.asarray(values)
        valid_host = np.asarray(valid_mask, dtype=bool)
        status_host = np.asarray(status, dtype=np.int32)
        available_host = np.asarray(availability_time_ns, dtype=np.int64)
        if event_host.ndim != 1 or decision_host.shape != event_host.shape:
            raise ValueError(
                "panel event and decision times must be equal-length vectors."
            )
        shape = (event_host.size, layout.factor_count)
        if value_host.shape != shape or value_host.dtype.kind not in "fiu":
            raise ValueError("panel values must have shape (time_count, factor_count).")
        if valid_host.shape != shape or status_host.shape != shape:
            raise ValueError("panel validity and status must match values.")
        if available_host.shape != shape:
            raise ValueError("panel availability times must match values.")
        identifiers = tuple(tuple(str(item) for item in row) for row in observation_ids)
        if len(identifiers) != shape[0] or any(
            len(row) != shape[1] for row in identifiers
        ):
            raise ValueError("observation_ids must match the panel shape.")
        if np.any(valid_host & (available_host > decision_host[:, None])):
            raise ValueError("A valid panel cell must be available by its decision time.")
        self.layout = layout
        self.event_time_ns = jnp.asarray(event_host)
        self.decision_time_ns = jnp.asarray(decision_host)
        self.values = jnp.asarray(value_host)
        self.valid_mask = jnp.asarray(valid_host)
        self.status = jnp.asarray(status_host)
        self.availability_time_ns = jnp.asarray(available_host)
        self.observation_ids = identifiers
        self.panel_id = canonical_fingerprint(
            {
                "kind": "financial-point-in-time-panel",
                "layout": layout.layout_id,
                "event_time_ns": event_host.tolist(),
                "decision_time_ns": decision_host.tolist(),
                "values": value_host.tolist(),
                "valid_mask": valid_host.tolist(),
                "status": status_host.tolist(),
                "availability_time_ns": available_host.tolist(),
                "observation_ids": [list(row) for row in identifiers],
            }
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: MarketDataSnapshot,
        layout: RiskFactorLayout,
        event_times: Sequence[FinancialTimestamp],
        decision_times: Sequence[FinancialTimestamp],
        /,
        *,
        tie_policy: QuoteTiePolicy = QuoteTiePolicy.REJECT,
    ) -> PointInTimePanel:
        """Build exact ``event_time`` cells using only data available at each decision."""

        if not isinstance(snapshot, MarketDataSnapshot):
            raise TypeError("snapshot must be a MarketDataSnapshot.")
        if not isinstance(layout, RiskFactorLayout):
            raise TypeError("layout must be a RiskFactorLayout.")
        events = tuple(event_times)
        decisions = tuple(decision_times)
        if not events or len(events) != len(decisions):
            raise ValueError(
                "event_times and decision_times must be equal non-empty sequences."
            )
        if not all(isinstance(value, FinancialTimestamp) for value in events + decisions):
            raise TypeError("panel times must be FinancialTimestamp values.")
        if not isinstance(tie_policy, QuoteTiePolicy):
            raise TypeError("tie_policy must be a QuoteTiePolicy.")
        event_ns = np.asarray(
            [value.epoch_nanoseconds for value in events], dtype=np.int64
        )
        decision_ns = np.asarray(
            [value.epoch_nanoseconds for value in decisions], dtype=np.int64
        )
        shape = (len(events), layout.factor_count)
        values = np.zeros(shape, dtype=float)
        valid = np.zeros(shape, dtype=bool)
        status = np.zeros(shape, dtype=np.int32)
        available = np.zeros(shape, dtype=np.int64)
        identifiers = [[""] * shape[1] for _ in range(shape[0])]
        by_key: dict[str, list[QuoteObservation]] = {}
        for observation in snapshot.observations:
            by_key.setdefault(observation.key.key_id, []).append(observation)
        for row, (event_clock, decision_clock) in enumerate(
            zip(event_ns, decision_ns, strict=True)
        ):
            for column, factor in enumerate(layout.keys):
                exact = [
                    item
                    for item in by_key.get(factor.quote_key.key_id, [])
                    if item.event_time_ns == int(event_clock)
                ]
                candidates = [
                    item
                    for item in exact
                    if item.available_time_ns <= int(decision_clock)
                ]
                if not candidates:
                    status[row, column] = int(
                        MarketStatus.CAUSAL_TIME_VIOLATION
                        if exact
                        else MarketStatus.MISSING_FACTOR
                    )
                    continue
                latest_available = max(item.available_time_ns for item in candidates)
                newest = [
                    item
                    for item in candidates
                    if item.available_time_ns == latest_available
                ]
                if len(newest) > 1 and tie_policy is QuoteTiePolicy.REJECT:
                    status[row, column] = int(MarketStatus.DUPLICATE)
                    continue
                if tie_policy is QuoteTiePolicy.EARLIEST_VINTAGE:
                    selected = min(
                        newest, key=lambda item: (item.vintage_id, item.observation_id)
                    )
                else:
                    selected = max(
                        newest, key=lambda item: (item.vintage_id, item.observation_id)
                    )
                number = float(np.asarray(selected.value))
                if np.isfinite(number):
                    values[row, column] = number
                    valid[row, column] = True
                else:
                    status[row, column] = int(MarketStatus.NONFINITE)
                available[row, column] = selected.available_time_ns
                identifiers[row][column] = selected.observation_id
        return cls(
            layout,
            event_ns,
            decision_ns,
            values,
            valid,
            status,
            available,
            observation_ids=identifiers,
        )

    @property
    def accepted(self) -> Array:
        return self.valid_mask & (self.status == int(MarketStatus.SUCCESS))


class MarketReplay(StrictModule):
    """Fixed-shape causal states at a sequence of decision clocks."""

    layout: RiskFactorLayout
    decision_time_ns: Array
    values: Array
    valid_mask: Array
    status: Array
    event_time_ns: Array
    availability_time_ns: Array
    replay_id: str = eqx.field(static=True)


class MarketEventStream(StrictModule, NonTrainableState):
    """Availability-ordered immutable event stream retaining every correction."""

    observations: tuple[QuoteObservation, ...]
    archive_time: FinancialTimestamp = eqx.field(static=True)
    stream_id: str = eqx.field(static=True)

    def __init__(
        self,
        observations: Sequence[QuoteObservation],
        /,
        *,
        archive_time: FinancialTimestamp,
    ):
        if not isinstance(archive_time, FinancialTimestamp):
            raise TypeError("archive_time must be a FinancialTimestamp.")
        values = tuple(observations)
        if not values or not all(isinstance(value, QuoteObservation) for value in values):
            raise TypeError("observations must contain at least one QuoteObservation.")
        if any(
            value.available_time_ns > archive_time.epoch_nanoseconds for value in values
        ):
            raise ValueError("An event stream cannot contain data beyond archive_time.")
        ordered = tuple(
            sorted(
                values,
                key=lambda value: (
                    value.available_time_ns,
                    value.event_time_ns,
                    value.key.key_id,
                    value.vintage_id,
                    value.observation_id,
                ),
            )
        )
        self.observations = ordered
        self.archive_time = archive_time
        self.stream_id = canonical_fingerprint(
            {
                "kind": "financial-market-event-stream",
                "observations": [value.observation_id for value in ordered],
                "archive_time_ns": int(archive_time.epoch_nanoseconds),
            }
        )

    def prepare(
        self,
        layout: RiskFactorLayout,
        /,
        *,
        capacity: int,
        tie_policy: QuoteTiePolicy = QuoteTiePolicy.REJECT,
        max_age_ns: int | None = None,
    ) -> PreparedMarketEventStream:
        return PreparedMarketEventStream(
            self,
            layout,
            capacity=capacity,
            tie_policy=tie_policy,
            max_age_ns=max_age_ns,
        )


class PreparedMarketEventStream(StrictModule, NonTrainableState):
    """Fixed-capacity event buffer plus deterministic causal replay route."""

    stream: MarketEventStream
    layout: RiskFactorLayout
    factor_index: Array
    values: Array
    event_time_ns: Array
    availability_time_ns: Array
    valid_mask: Array
    preparation_status: Array
    capacity: int = eqx.field(static=True)
    tie_policy: QuoteTiePolicy = eqx.field(static=True)
    max_age_ns: int | None = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        stream: MarketEventStream,
        layout: RiskFactorLayout,
        /,
        *,
        capacity: int,
        tie_policy: QuoteTiePolicy,
        max_age_ns: int | None,
    ):
        if not isinstance(stream, MarketEventStream):
            raise TypeError("stream must be a MarketEventStream.")
        if not isinstance(layout, RiskFactorLayout):
            raise TypeError("layout must be a RiskFactorLayout.")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer.")
        if not isinstance(tie_policy, QuoteTiePolicy):
            raise TypeError("tie_policy must be a QuoteTiePolicy.")
        if max_age_ns is not None and (
            isinstance(max_age_ns, bool)
            or not isinstance(max_age_ns, int)
            or max_age_ns < 0
        ):
            raise ValueError("max_age_ns must be a nonnegative integer or None.")
        count = min(len(stream.observations), capacity)
        selected = stream.observations[:count]
        factor_lookup = {
            factor.quote_key.key_id: index for index, factor in enumerate(layout.keys)
        }
        factor_index = np.full((capacity,), -1, dtype=np.int32)
        values = np.zeros((capacity,), dtype=float)
        event = np.zeros((capacity,), dtype=np.int64)
        available = np.zeros((capacity,), dtype=np.int64)
        valid = np.zeros((capacity,), dtype=bool)
        for index, observation in enumerate(selected):
            factor_index[index] = factor_lookup.get(observation.key.key_id, -1)
            values[index] = float(np.asarray(observation.value))
            event[index] = observation.event_time_ns
            available[index] = observation.available_time_ns
            valid[index] = factor_index[index] >= 0 and np.isfinite(values[index])
        overflow = len(stream.observations) > capacity
        preparation_status = (
            MarketStatus.CAPACITY_EXCEEDED if overflow else MarketStatus.SUCCESS
        )
        self.stream = stream
        self.layout = layout
        self.factor_index = jnp.asarray(factor_index)
        self.values = jnp.asarray(values)
        self.event_time_ns = jnp.asarray(event)
        self.availability_time_ns = jnp.asarray(available)
        self.valid_mask = jnp.asarray(valid)
        self.preparation_status = jnp.asarray(int(preparation_status), dtype=jnp.int32)
        self.capacity = capacity
        self.tie_policy = tie_policy
        self.max_age_ns = max_age_ns
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-financial-market-event-stream",
                "stream": stream.stream_id,
                "layout": layout.layout_id,
                "capacity": capacity,
                "tie_policy": tie_policy.value,
                "max_age_ns": max_age_ns,
                "factor_index": factor_index.tolist(),
                "valid_mask": valid.tolist(),
                "status": int(preparation_status),
            }
        )

    def replay(self, decision_times: Sequence[FinancialTimestamp], /) -> MarketReplay:
        decisions = tuple(decision_times)
        if not decisions or not all(
            isinstance(value, FinancialTimestamp) for value in decisions
        ):
            raise TypeError("decision_times must contain FinancialTimestamp values.")
        snapshot = MarketDataSnapshot(
            self.stream.observations,
            snapshot_time=self.stream.archive_time,
        )
        states = tuple(
            snapshot.prepare(
                self.layout,
                decision,
                tie_policy=self.tie_policy,
                max_age_ns=self.max_age_ns,
            )
            for decision in decisions
        )
        values = jnp.stack(tuple(state.values for state in states))
        valid = jnp.stack(tuple(state.valid_mask for state in states))
        status = jnp.stack(tuple(state.status for state in states))
        if int(np.asarray(self.preparation_status)) != int(MarketStatus.SUCCESS):
            valid = jnp.zeros_like(valid)
            status = status | self.preparation_status
        event = jnp.stack(tuple(state.event_time_ns for state in states))
        available = jnp.stack(tuple(state.availability_time_ns for state in states))
        clocks = jnp.asarray(
            [value.epoch_nanoseconds for value in decisions], dtype=jnp.int64
        )
        replay_id = canonical_fingerprint(
            {
                "kind": "financial-market-replay",
                "prepared": self.prepared_id,
                "decision_time_ns": np.asarray(clocks).tolist(),
            }
        )
        return MarketReplay(
            self.layout,
            clocks,
            values,
            valid,
            status,
            event,
            available,
            replay_id,
        )


__all__ = [
    "MarketEventStream",
    "MarketReplay",
    "PointInTimePanel",
    "PreparedMarketEventStream",
]
