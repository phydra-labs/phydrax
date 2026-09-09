#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..core import FinancialTimestamp
from ..market import MarketDataSnapshot, QuoteKey, QuoteObservation, ReferenceDataSnapshot


PanelClock: TypeAlias = Literal["event", "published", "received", "available"]


class PointInTimePanelDefinition(StrictModule):
    """Host-resolved quote panel whose information cutoff is explicit."""

    quote_keys: tuple[QuoteKey, ...] = eqx.field(static=True)
    analysis_time: FinancialTimestamp = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    clock: PanelClock = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        quote_keys: Sequence[QuoteKey],
        analysis_time: FinancialTimestamp,
        /,
        *,
        capacity: int,
        clock: PanelClock = "event",
    ):
        keys = tuple(quote_keys)
        if not keys or not all(isinstance(key, QuoteKey) for key in keys):
            raise TypeError("quote_keys must be a nonempty sequence of QuoteKey values.")
        if len({key.key_id for key in keys}) != len(keys):
            raise ValueError("quote_keys must be unique and canonically ordered.")
        if tuple(sorted(key.key_id for key in keys)) != tuple(key.key_id for key in keys):
            raise ValueError("quote_keys must be ordered by key_id.")
        if not isinstance(analysis_time, FinancialTimestamp):
            raise TypeError("analysis_time must be a FinancialTimestamp.")
        capacity_ = int(capacity)
        if capacity_ < 1:
            raise ValueError("capacity must be positive.")
        if clock not in ("event", "published", "received", "available"):
            raise ValueError("clock must name one of the four preserved clocks.")
        self.quote_keys = keys
        self.analysis_time = analysis_time
        self.capacity = capacity_
        self.clock = clock
        self.definition_id = canonical_fingerprint(
            {
                "kind": "point-in-time-panel-definition",
                "keys": [key.key_id for key in keys],
                "analysis_time_ns": int(analysis_time.epoch_nanoseconds),
                "capacity": capacity_,
                "clock": clock,
            }
        )


class ResolvedPointInTimePanel(StrictModule):
    """Causally selected host observations before fixed-shape lowering."""

    definition: PointInTimePanelDefinition = eqx.field(static=True)
    observations: tuple[tuple[QuoteObservation, ...], ...]
    future_revision_counts: tuple[int, ...] = eqx.field(static=True)
    reference_data_id: str = eqx.field(static=True)
    market_data_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)


class PreparedPointInTimePanel(StrictModule):
    """Fixed-shape device panel retaining event, publication, receipt, and availability."""

    values: Array
    valid_mask: Array
    event_times_ns: Array
    published_times_ns: Array
    received_times_ns: Array
    available_times_ns: Array
    observation_ids: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    vintage_ids: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    quote_key_ids: tuple[str, ...] = eqx.field(static=True)
    analysis_time_ns: int = eqx.field(static=True)
    clock: PanelClock = eqx.field(static=True)
    reference_data_id: str = eqx.field(static=True)
    market_data_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capacity: int = eqx.field(static=True)

    @property
    def series_count(self) -> int:
        return len(self.quote_key_ids)


class PanelAvailabilityEvidence(StrictModule):
    """Data/convention evidence for one point-in-time resolution or replay."""

    active_counts: Array
    masked_counts: Array
    future_revision_count: Array
    overflow: Array
    clocks_admissible: Array
    replay_equal: Array
    reference_data_id: str = eqx.field(static=True)
    market_data_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class MarketEventStreamDefinition(StrictModule):
    """Flat market-event stream selected by quote channels and knowledge time."""

    quote_keys: tuple[QuoteKey, ...] = eqx.field(static=True)
    analysis_time: FinancialTimestamp = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    ordering_clock: PanelClock = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        quote_keys: Sequence[QuoteKey],
        analysis_time: FinancialTimestamp,
        /,
        *,
        capacity: int,
        ordering_clock: PanelClock = "available",
    ):
        panel = PointInTimePanelDefinition(
            quote_keys,
            analysis_time,
            capacity=capacity,
            clock=ordering_clock,
        )
        self.quote_keys = panel.quote_keys
        self.analysis_time = analysis_time
        self.capacity = panel.capacity
        self.ordering_clock = ordering_clock
        self.definition_id = canonical_fingerprint(
            {
                "kind": "market-event-stream-definition",
                "keys": [key.key_id for key in panel.quote_keys],
                "analysis_time_ns": int(analysis_time.epoch_nanoseconds),
                "capacity": panel.capacity,
                "ordering_clock": ordering_clock,
            }
        )


class PreparedMarketEventStream(StrictModule):
    """Fixed-capacity event stream with deterministic channel and tie ordering."""

    values: Array
    channels: Array
    valid_mask: Array
    event_times_ns: Array
    published_times_ns: Array
    received_times_ns: Array
    available_times_ns: Array
    observation_ids: tuple[str, ...] = eqx.field(static=True)
    vintage_ids: tuple[str, ...] = eqx.field(static=True)
    quote_key_ids: tuple[str, ...] = eqx.field(static=True)
    analysis_time_ns: int = eqx.field(static=True)
    ordering_clock: PanelClock = eqx.field(static=True)
    overflow: bool = eqx.field(static=True)
    stream_id: str = eqx.field(static=True)
    capacity: int = eqx.field(static=True)


def _clock(observation: QuoteObservation, name: PanelClock) -> int:
    if name == "event":
        return int(observation.timestamp.epoch_nanoseconds)
    if name == "published":
        return int(observation.timestamp.published_ns)
    if name == "received":
        return int(observation.timestamp.received_ns)
    return int(observation.timestamp.available_ns)


def _reference_ids(snapshot: ReferenceDataSnapshot) -> set[str]:
    assets = {
        f"{asset.identifier.scheme}:{asset.identifier.value}" for asset in snapshot.assets
    }
    instruments = {
        f"{instrument.identifier.scheme}:{instrument.identifier.value}"
        for instrument in snapshot.instruments
    }
    return assets | instruments


def resolve_point_in_time_panel(
    definition: PointInTimePanelDefinition,
    reference_data: ReferenceDataSnapshot,
    market_data: MarketDataSnapshot,
    /,
) -> ResolvedPointInTimePanel:
    """Select the latest causally available vintage of each effective observation."""

    if not isinstance(definition, PointInTimePanelDefinition):
        raise TypeError("definition must be a PointInTimePanelDefinition.")
    if not isinstance(reference_data, ReferenceDataSnapshot):
        raise TypeError("reference_data must be a ReferenceDataSnapshot.")
    if not isinstance(market_data, MarketDataSnapshot):
        raise TypeError("market_data must be a MarketDataSnapshot.")
    if (
        market_data.reference_data_id
        and market_data.reference_data_id != reference_data.snapshot_id
    ):
        raise ValueError("market_data is bound to a different reference-data snapshot.")
    known = _reference_ids(reference_data)
    if any(key.reference_id not in known for key in definition.quote_keys):
        raise ValueError("every panel quote key must reference the supplied universe.")
    cutoff = int(definition.analysis_time.available_ns)
    key_ids = {key.key_id for key in definition.quote_keys}
    grouped: dict[tuple[str, int], list[QuoteObservation]] = {}
    future_counts = {key.key_id: 0 for key in definition.quote_keys}
    for observation in market_data.observations:
        if observation.key.key_id not in key_ids:
            continue
        if observation.available_time_ns > cutoff:
            future_counts[observation.key.key_id] += 1
            continue
        grouped.setdefault(
            (observation.key.key_id, observation.event_time_ns), []
        ).append(observation)
    selected: dict[str, list[QuoteObservation]] = {
        key.key_id: [] for key in definition.quote_keys
    }
    for (key_id, _), candidates in grouped.items():
        chosen = max(
            candidates,
            key=lambda value: (
                value.available_time_ns,
                int(value.timestamp.received_ns),
                value.vintage_id,
                value.observation_id,
            ),
        )
        selected[key_id].append(chosen)
    rows = tuple(
        tuple(
            sorted(
                selected[key.key_id],
                key=lambda value: (
                    _clock(value, definition.clock),
                    value.event_time_ns,
                    value.available_time_ns,
                    value.observation_id,
                ),
            )
        )
        for key in definition.quote_keys
    )
    resolved_id = canonical_fingerprint(
        {
            "kind": "resolved-point-in-time-panel",
            "definition": definition.definition_id,
            "reference_data": reference_data.snapshot_id,
            "market_data": market_data.snapshot_id,
            "observations": [
                [observation.observation_id for observation in row] for row in rows
            ],
        }
    )
    return ResolvedPointInTimePanel(
        definition=definition,
        observations=rows,
        future_revision_counts=tuple(
            future_counts[key.key_id] for key in definition.quote_keys
        ),
        reference_data_id=reference_data.snapshot_id,
        market_data_id=market_data.snapshot_id,
        resolved_id=resolved_id,
    )


def prepare_point_in_time_panel(
    resolved: ResolvedPointInTimePanel,
    /,
) -> tuple[PreparedPointInTimePanel, PanelAvailabilityEvidence]:
    """Lower resolved host records into deterministic fixed-capacity device arrays."""

    if not isinstance(resolved, ResolvedPointInTimePanel):
        raise TypeError("resolved must be a ResolvedPointInTimePanel.")
    definition = resolved.definition
    series_count = len(definition.quote_keys)
    capacity = definition.capacity
    values = np.zeros((series_count, capacity), dtype=float)
    valid = np.zeros((series_count, capacity), dtype=bool)
    event = np.zeros((series_count, capacity), dtype=np.int64)
    published = np.zeros((series_count, capacity), dtype=np.int64)
    received = np.zeros((series_count, capacity), dtype=np.int64)
    available = np.zeros((series_count, capacity), dtype=np.int64)
    observation_ids: list[tuple[str, ...]] = []
    vintage_ids: list[tuple[str, ...]] = []
    overflow = np.zeros((series_count,), dtype=bool)
    admissible = np.ones((series_count,), dtype=bool)
    for series, observations in enumerate(resolved.observations):
        overflow[series] = len(observations) > capacity
        chosen = observations[:capacity]
        ids = [""] * capacity
        vintages = [""] * capacity
        for index, observation in enumerate(chosen):
            timestamp = observation.timestamp
            number = float(np.asarray(observation.value))
            if np.isfinite(number):
                values[series, index] = number
                valid[series, index] = True
            event[series, index] = int(timestamp.epoch_nanoseconds)
            published[series, index] = int(timestamp.published_ns)
            received[series, index] = int(timestamp.received_ns)
            available[series, index] = int(timestamp.available_ns)
            ids[index] = observation.observation_id
            vintages[index] = observation.vintage_id
            admissible[series] &= (
                int(timestamp.published_ns)
                <= int(timestamp.received_ns)
                <= int(timestamp.available_ns)
                <= int(definition.analysis_time.available_ns)
            )
        observation_ids.append(tuple(ids))
        vintage_ids.append(tuple(vintages))
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-point-in-time-panel",
            "resolved": resolved.resolved_id,
            "observation_ids": observation_ids,
            "capacity": capacity,
        }
    )
    prepared = PreparedPointInTimePanel(
        values=jnp.asarray(values),
        valid_mask=jnp.asarray(valid),
        event_times_ns=jnp.asarray(event),
        published_times_ns=jnp.asarray(published),
        received_times_ns=jnp.asarray(received),
        available_times_ns=jnp.asarray(available),
        observation_ids=tuple(observation_ids),
        vintage_ids=tuple(vintage_ids),
        quote_key_ids=tuple(key.key_id for key in definition.quote_keys),
        analysis_time_ns=int(definition.analysis_time.available_ns),
        clock=definition.clock,
        reference_data_id=resolved.reference_data_id,
        market_data_id=resolved.market_data_id,
        resolved_id=resolved.resolved_id,
        prepared_id=prepared_id,
        capacity=capacity,
    )
    evidence = PanelAvailabilityEvidence(
        active_counts=jnp.sum(prepared.valid_mask, axis=-1).astype(jnp.int32),
        masked_counts=jnp.sum(~prepared.valid_mask, axis=-1).astype(jnp.int32),
        future_revision_count=jnp.asarray(
            resolved.future_revision_counts, dtype=jnp.int32
        ),
        overflow=jnp.asarray(overflow),
        clocks_admissible=jnp.asarray(admissible),
        replay_equal=jnp.asarray(True),
        reference_data_id=resolved.reference_data_id,
        market_data_id=resolved.market_data_id,
        definition_id=definition.definition_id,
        resolved_id=resolved.resolved_id,
        prepared_id=prepared_id,
    )
    return prepared, evidence


def replay_market_data(
    prepared: PreparedPointInTimePanel,
    definition: PointInTimePanelDefinition,
    reference_data: ReferenceDataSnapshot,
    market_data: MarketDataSnapshot,
    /,
) -> PanelAvailabilityEvidence:
    """Independently rebuild and exactly compare a prepared point-in-time panel."""

    if not isinstance(prepared, PreparedPointInTimePanel):
        raise TypeError("prepared must be a PreparedPointInTimePanel.")
    replayed, evidence = prepare_point_in_time_panel(
        resolve_point_in_time_panel(definition, reference_data, market_data)
    )
    equal = (
        replayed.prepared_id == prepared.prepared_id
        and replayed.observation_ids == prepared.observation_ids
        and bool(jnp.array_equal(replayed.valid_mask, prepared.valid_mask))
        and bool(jnp.array_equal(replayed.values, prepared.values))
        and bool(
            jnp.array_equal(replayed.available_times_ns, prepared.available_times_ns)
        )
    )
    return PanelAvailabilityEvidence(
        active_counts=evidence.active_counts,
        masked_counts=evidence.masked_counts,
        future_revision_count=evidence.future_revision_count,
        overflow=evidence.overflow,
        clocks_admissible=evidence.clocks_admissible,
        replay_equal=jnp.asarray(equal),
        reference_data_id=evidence.reference_data_id,
        market_data_id=evidence.market_data_id,
        definition_id=evidence.definition_id,
        resolved_id=evidence.resolved_id,
        prepared_id=evidence.prepared_id,
    )


def prepare_market_event_stream(
    definition: MarketEventStreamDefinition,
    market_data: MarketDataSnapshot,
    /,
) -> PreparedMarketEventStream:
    """Prepare every causally admissible quote observation as one ordered event stream."""

    if not isinstance(definition, MarketEventStreamDefinition):
        raise TypeError("definition must be a MarketEventStreamDefinition.")
    if not isinstance(market_data, MarketDataSnapshot):
        raise TypeError("market_data must be a MarketDataSnapshot.")
    key_to_channel = {
        key.key_id: index for index, key in enumerate(definition.quote_keys)
    }
    cutoff = int(definition.analysis_time.available_ns)
    selected = [
        observation
        for observation in market_data.observations
        if observation.key.key_id in key_to_channel
        and observation.available_time_ns <= cutoff
    ]
    selected.sort(
        key=lambda value: (
            _clock(value, definition.ordering_clock),
            value.available_time_ns,
            int(value.timestamp.received_ns),
            value.event_time_ns,
            value.vintage_id,
            value.observation_id,
        )
    )
    capacity = definition.capacity
    overflow = len(selected) > capacity
    chosen = selected[:capacity]
    values = np.zeros((capacity,), dtype=float)
    channels = np.zeros((capacity,), dtype=np.int32)
    valid = np.zeros((capacity,), dtype=bool)
    clocks = [np.zeros((capacity,), dtype=np.int64) for _ in range(4)]
    observation_ids = [""] * capacity
    vintage_ids = [""] * capacity
    for index, observation in enumerate(chosen):
        timestamp = observation.timestamp
        values[index] = float(np.asarray(observation.value))
        channels[index] = key_to_channel[observation.key.key_id]
        valid[index] = True
        clocks[0][index] = int(timestamp.epoch_nanoseconds)
        clocks[1][index] = int(timestamp.published_ns)
        clocks[2][index] = int(timestamp.received_ns)
        clocks[3][index] = int(timestamp.available_ns)
        observation_ids[index] = observation.observation_id
        vintage_ids[index] = observation.vintage_id
    stream_id = canonical_fingerprint(
        {
            "kind": "prepared-market-event-stream",
            "definition": definition.definition_id,
            "market_data": market_data.snapshot_id,
            "observations": observation_ids,
        }
    )
    return PreparedMarketEventStream(
        values=jnp.asarray(values),
        channels=jnp.asarray(channels),
        valid_mask=jnp.asarray(valid),
        event_times_ns=jnp.asarray(clocks[0]),
        published_times_ns=jnp.asarray(clocks[1]),
        received_times_ns=jnp.asarray(clocks[2]),
        available_times_ns=jnp.asarray(clocks[3]),
        observation_ids=tuple(observation_ids),
        vintage_ids=tuple(vintage_ids),
        quote_key_ids=tuple(key.key_id for key in definition.quote_keys),
        analysis_time_ns=cutoff,
        ordering_clock=definition.ordering_clock,
        overflow=overflow,
        stream_id=stream_id,
        capacity=capacity,
    )


__all__ = [
    "MarketEventStreamDefinition",
    "PanelAvailabilityEvidence",
    "PanelClock",
    "PointInTimePanelDefinition",
    "PreparedMarketEventStream",
    "PreparedPointInTimePanel",
    "ResolvedPointInTimePanel",
    "prepare_market_event_stream",
    "prepare_point_in_time_panel",
    "replay_market_data",
    "resolve_point_in_time_panel",
]
