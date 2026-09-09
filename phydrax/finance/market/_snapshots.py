#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..core import (
    AssetReference,
    Currency,
    FinancialTimestamp,
    InstrumentReference,
)
from ._lineage import DataLineage
from ._quotes import QuoteObservation, QuoteTiePolicy
from ._risk_factors import MarketState, RiskFactorLayout
from ._status import MarketStatus


def _identifier_key(reference: AssetReference | InstrumentReference, /) -> str:
    return f"{reference.identifier.scheme}:{reference.identifier.value}"


def _snapshot_clock(value: FinancialTimestamp, name: str, /) -> int:
    if not isinstance(value, FinancialTimestamp):
        raise TypeError(f"{name} must be a FinancialTimestamp.")
    return int(value.epoch_nanoseconds)


class ReferenceDataSnapshot(StrictModule, NonTrainableState):
    """Immutable point-in-time financial reference-data universe."""

    assets: tuple[AssetReference, ...]
    instruments: tuple[InstrumentReference, ...]
    currencies: tuple[Currency, ...]
    as_of: FinancialTimestamp = eqx.field(static=True)
    lineage: DataLineage
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        assets: Sequence[AssetReference] = (),
        instruments: Sequence[InstrumentReference] = (),
        currencies: Sequence[Currency] = (),
        /,
        *,
        as_of: FinancialTimestamp,
        lineage: DataLineage,
    ):
        asset_values = tuple(assets)
        instrument_values = tuple(instruments)
        currency_values = tuple(currencies)
        if not all(isinstance(value, AssetReference) for value in asset_values):
            raise TypeError("assets must contain only AssetReference values.")
        if not all(isinstance(value, InstrumentReference) for value in instrument_values):
            raise TypeError("instruments must contain only InstrumentReference values.")
        if not all(isinstance(value, Currency) for value in currency_values):
            raise TypeError("currencies must contain only Currency values.")
        if not isinstance(as_of, FinancialTimestamp):
            raise TypeError("as_of must be a FinancialTimestamp.")
        if not isinstance(lineage, DataLineage):
            raise TypeError("lineage must be a DataLineage.")
        asset_ids = tuple(_identifier_key(value) for value in asset_values)
        instrument_ids = tuple(_identifier_key(value) for value in instrument_values)
        currency_codes = tuple(value.code for value in currency_values)
        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError("Reference-data asset identifiers must be unique.")
        if len(set(instrument_ids)) != len(instrument_ids):
            raise ValueError("Reference-data instrument identifiers must be unique.")
        if len(set(currency_codes)) != len(currency_codes):
            raise ValueError("Reference-data currency codes must be unique.")
        self.assets = tuple(sorted(asset_values, key=_identifier_key))
        self.instruments = tuple(sorted(instrument_values, key=_identifier_key))
        self.currencies = tuple(sorted(currency_values, key=lambda value: value.code))
        self.as_of = as_of
        self.lineage = lineage
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "financial-reference-data-snapshot",
                "asset_ids": sorted(asset_ids),
                "instrument_ids": sorted(instrument_ids),
                "currency_ids": sorted(value.currency_id for value in currency_values),
                "as_of_ns": int(as_of.epoch_nanoseconds),
                "as_of_available_ns": int(as_of.available_ns),
                "as_of_vintage_id": as_of.vintage_id,
                "lineage": lineage.lineage_id,
            }
        )

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        /,
        *,
        assets: Sequence[AssetReference],
        instruments: Sequence[InstrumentReference],
        currencies: Sequence[Currency],
        as_of: FinancialTimestamp,
        lineage: DataLineage,
    ) -> ReferenceDataSnapshot:
        value = cls(assets, instruments, currencies, as_of=as_of, lineage=lineage)
        if "snapshot_id" in record and str(record["snapshot_id"]) != value.snapshot_id:
            raise ValueError("Reference snapshot record does not match snapshot_id.")
        return value

    def to_record(self) -> Mapping[str, Any]:
        return {
            "asset_ids": [_identifier_key(value) for value in self.assets],
            "instrument_ids": [_identifier_key(value) for value in self.instruments],
            "currencies": [value.code for value in self.currencies],
            "as_of_ns": int(self.as_of.epoch_nanoseconds),
            "lineage_id": self.lineage.lineage_id,
            "snapshot_id": self.snapshot_id,
        }


class MarketDataSnapshot(StrictModule, NonTrainableState):
    """Immutable quote archive whose bitemporal vintages are never overwritten."""

    observations: tuple[QuoteObservation, ...]
    snapshot_time: FinancialTimestamp = eqx.field(static=True)
    reference_data_id: str = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        observations: Sequence[QuoteObservation],
        /,
        *,
        snapshot_time: FinancialTimestamp,
        reference_data_id: str = "",
    ):
        values = tuple(observations)
        if not all(isinstance(value, QuoteObservation) for value in values):
            raise TypeError("observations must contain QuoteObservation values.")
        if not isinstance(snapshot_time, FinancialTimestamp):
            raise TypeError("snapshot_time must be a FinancialTimestamp.")
        if not isinstance(reference_data_id, str):
            raise TypeError("reference_data_id must be a string.")
        snapshot_ns = int(snapshot_time.epoch_nanoseconds)
        if any(value.available_time_ns > snapshot_ns for value in values):
            raise ValueError(
                "A snapshot cannot contain observations unavailable at snapshot_time."
            )
        ordered = tuple(
            sorted(
                values,
                key=lambda value: (
                    value.key.key_id,
                    value.event_time_ns,
                    value.available_time_ns,
                    value.vintage_id,
                    value.observation_id,
                ),
            )
        )
        self.observations = ordered
        self.snapshot_time = snapshot_time
        self.reference_data_id = reference_data_id.strip()
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "financial-market-data-snapshot",
                "observations": [value.observation_id for value in ordered],
                "snapshot_available_ns": int(snapshot_time.available_ns),
                "snapshot_vintage_id": snapshot_time.vintage_id,
                "snapshot_time_ns": snapshot_ns,
                "reference_data_id": self.reference_data_id,
            }
        )

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        /,
        *,
        observations: Sequence[QuoteObservation],
        snapshot_time: FinancialTimestamp,
    ) -> MarketDataSnapshot:
        value = cls(
            observations,
            snapshot_time=snapshot_time,
            reference_data_id=str(record["reference_data_id"]),
        )
        if "snapshot_id" in record and str(record["snapshot_id"]) != value.snapshot_id:
            raise ValueError("Market snapshot record does not match snapshot_id.")
        return value

    def prepare(
        self,
        layout: RiskFactorLayout,
        decision_time: FinancialTimestamp,
        /,
        *,
        tie_policy: QuoteTiePolicy = QuoteTiePolicy.REJECT,
        max_age_ns: int | None = None,
        capacity: int | None = None,
    ) -> MarketState:
        """Resolve one causal host snapshot into fixed-shape device coordinates."""

        if not isinstance(layout, RiskFactorLayout):
            raise TypeError("layout must be a RiskFactorLayout.")
        decision_ns = _snapshot_clock(decision_time, "decision_time")
        if not isinstance(tie_policy, QuoteTiePolicy):
            raise TypeError("tie_policy must be a QuoteTiePolicy.")
        if max_age_ns is not None and (
            isinstance(max_age_ns, bool)
            or not isinstance(max_age_ns, int)
            or max_age_ns < 0
        ):
            raise ValueError("max_age_ns must be a nonnegative integer or None.")
        if capacity is not None and (
            isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1
        ):
            raise ValueError("capacity must be a positive integer or None.")
        count = layout.factor_count
        values = np.zeros((count,), dtype=float)
        valid = np.zeros((count,), dtype=bool)
        status = np.zeros((count,), dtype=np.int32)
        event = np.zeros((count,), dtype=np.int64)
        available = np.zeros((count,), dtype=np.int64)
        observation_ids = [""] * count
        if capacity is not None and count > capacity:
            status[:] = int(MarketStatus.CAPACITY_EXCEEDED)
            return MarketState(
                layout,
                values,
                valid,
                status,
                event,
                available,
                decision_time_ns=decision_ns,
                observation_ids=observation_ids,
            )
        by_key: dict[str, list[QuoteObservation]] = {}
        for observation in self.observations:
            by_key.setdefault(observation.key.key_id, []).append(observation)
        for index, factor in enumerate(layout.keys):
            all_key = by_key.get(factor.quote_key.key_id, [])
            effective = [item for item in all_key if item.event_time_ns <= decision_ns]
            candidates = [
                item for item in effective if item.available_time_ns <= decision_ns
            ]
            if not candidates:
                status[index] = int(
                    MarketStatus.CAUSAL_TIME_VIOLATION
                    if effective
                    else MarketStatus.MISSING_FACTOR
                )
                continue
            latest_event = max(item.event_time_ns for item in candidates)
            event_candidates = [
                item for item in candidates if item.event_time_ns == latest_event
            ]
            latest_available = max(item.available_time_ns for item in event_candidates)
            newest = [
                item
                for item in event_candidates
                if item.available_time_ns == latest_available
            ]
            if len(newest) > 1 and tie_policy is QuoteTiePolicy.REJECT:
                status[index] = int(MarketStatus.DUPLICATE)
                continue
            if tie_policy is QuoteTiePolicy.EARLIEST_VINTAGE:
                selected = min(
                    newest, key=lambda item: (item.vintage_id, item.observation_id)
                )
            else:
                selected = max(
                    newest, key=lambda item: (item.vintage_id, item.observation_id)
                )
            selected_value = float(np.asarray(selected.value))
            selected_status = MarketStatus.SUCCESS
            if not np.isfinite(selected_value):
                selected_status |= MarketStatus.NONFINITE
            if (
                max_age_ns is not None
                and decision_ns - selected.event_time_ns > max_age_ns
            ):
                selected_status |= MarketStatus.STALE
            values[index] = selected_value if np.isfinite(selected_value) else 0.0
            event[index] = selected.event_time_ns
            available[index] = selected.available_time_ns
            observation_ids[index] = selected.observation_id
            status[index] = int(selected_status)
            valid[index] = selected_status == MarketStatus.SUCCESS
        _mark_crossed_quotes(layout, values, valid, status)
        return MarketState(
            layout,
            values,
            valid,
            status,
            event,
            available,
            decision_time_ns=decision_ns,
            observation_ids=observation_ids,
        )

    def to_record(self) -> Mapping[str, Any]:
        return {
            "observation_ids": [value.observation_id for value in self.observations],
            "snapshot_time_ns": int(self.snapshot_time.epoch_nanoseconds),
            "reference_data_id": self.reference_data_id,
            "snapshot_id": self.snapshot_id,
        }


def _mark_crossed_quotes(
    layout: RiskFactorLayout,
    values: np.ndarray,
    valid: np.ndarray,
    status: np.ndarray,
    /,
) -> None:
    sides: dict[tuple[str, str, str, str], dict[str, int]] = {}
    for index, factor in enumerate(layout.keys):
        quote = factor.quote_key
        if quote.field in ("bid", "ask"):
            currency = "" if quote.currency is None else quote.currency.currency_id
            pair = "" if quote.fx_pair is None else quote.fx_pair.pair_id
            sides.setdefault((quote.reference_id, quote.venue, currency, pair), {})[
                quote.field
            ] = index
    unavailable = int(
        MarketStatus.DUPLICATE
        | MarketStatus.MISSING_FACTOR
        | MarketStatus.CAUSAL_TIME_VIOLATION
        | MarketStatus.NONFINITE
        | MarketStatus.CAPACITY_EXCEEDED
    )
    for side in sides.values():
        if "bid" not in side or "ask" not in side:
            continue
        bid = side["bid"]
        ask = side["ask"]
        selected = (status[bid] & unavailable) == 0 and (status[ask] & unavailable) == 0
        if selected and values[bid] > values[ask]:
            status[bid] |= int(MarketStatus.CROSSED_QUOTE)
            status[ask] |= int(MarketStatus.CROSSED_QUOTE)
            valid[bid] = False
            valid[ask] = False


__all__ = ["MarketDataSnapshot", "ReferenceDataSnapshot"]
