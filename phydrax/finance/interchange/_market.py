#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core import Currency, FinancialTimestamp, FXPair, TemporalAdmissibilityPolicy
from ..market import DataLineage, MarketDataSnapshot, QuoteKey, QuoteObservation
from ._records import FinanceRecordBatch


def _timestamp_record(value: FinancialTimestamp, /) -> dict[str, object]:
    if not isinstance(value, FinancialTimestamp):
        raise TypeError("value must be a FinancialTimestamp.")
    return {
        "event_time_ns": value.event_ns,
        "published_time_ns": value.published_ns,
        "received_time_ns": value.received_ns,
        "available_time_ns": value.available_ns,
        "vintage_id": value.vintage_id,
        "policy": {
            "require_published_before_received": (
                value.policy.require_published_before_received
            ),
            "require_received_before_available": (
                value.policy.require_received_before_available
            ),
            "allow_future_effective_event": value.policy.allow_future_effective_event,
            "policy_id": value.policy.policy_id,
        },
    }


def _timestamp_from_record(record: Mapping[str, Any], /) -> FinancialTimestamp:
    policy_record = record["policy"]
    if not isinstance(policy_record, Mapping):
        raise TypeError("Serialized timestamp policy must be a mapping.")
    expected_policy = {
        "require_published_before_received",
        "require_received_before_available",
        "allow_future_effective_event",
        "policy_id",
    }
    if set(policy_record) != expected_policy:
        raise ValueError("Serialized timestamp policy fields are not canonical.")
    policy = TemporalAdmissibilityPolicy(
        policy_record["require_published_before_received"],
        policy_record["require_received_before_available"],
        policy_record["allow_future_effective_event"],
    )
    if policy.policy_id != policy_record["policy_id"]:
        raise ValueError("Serialized temporal policy identity is invalid.")
    expected = {
        "event_time_ns",
        "published_time_ns",
        "received_time_ns",
        "available_time_ns",
        "vintage_id",
        "policy",
    }
    if set(record) != expected:
        raise ValueError("Serialized financial timestamp fields are not canonical.")
    return FinancialTimestamp(
        record["event_time_ns"],
        record["published_time_ns"],
        record["received_time_ns"],
        record["available_time_ns"],
        record["vintage_id"],
        policy,
    )


def _currency_record(value: Currency | None, /) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "code": value.code,
        "minor_unit": value.minor_unit,
        "currency_id": value.currency_id,
    }


def _currency_from_record(record: object, /) -> Currency | None:
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise TypeError("Serialized currency must be a mapping or null.")
    if set(record) != {"code", "minor_unit", "currency_id"}:
        raise ValueError("Serialized currency fields are not canonical.")
    value = Currency(record["code"], record["minor_unit"])
    if value.currency_id != record["currency_id"]:
        raise ValueError("Serialized currency identity is invalid.")
    return value


def _pair_record(value: FXPair | None, /) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "base": _currency_record(value.base),
        "quote": _currency_record(value.quote),
        "pair_id": value.pair_id,
    }


def _pair_from_record(record: object, /) -> FXPair | None:
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise TypeError("Serialized FX pair must be a mapping or null.")
    if set(record) != {"base", "quote", "pair_id"}:
        raise ValueError("Serialized FX-pair fields are not canonical.")
    base = _currency_from_record(record["base"])
    quote = _currency_from_record(record["quote"])
    if base is None or quote is None:
        raise ValueError("Serialized FX-pair currencies cannot be null.")
    value = FXPair(base, quote)
    if value.pair_id != record["pair_id"]:
        raise ValueError("Serialized FX-pair identity is invalid.")
    return value


def _key_record(value: QuoteKey, /) -> dict[str, object]:
    record = dict(value.to_record())
    record["currency"] = _currency_record(value.currency)
    record["fx_pair"] = _pair_record(value.fx_pair)
    return record


def _key_from_record(record: Mapping[str, Any], /) -> QuoteKey:
    expected = {
        "reference_id",
        "field",
        "venue",
        "currency",
        "fx_pair",
        "key_id",
    }
    if set(record) != expected:
        raise ValueError("Serialized quote-key fields are not canonical.")
    currency = _currency_from_record(record["currency"])
    pair = _pair_from_record(record["fx_pair"])
    compact = dict(record)
    compact["currency"] = None if currency is None else currency.code
    compact["fx_pair"] = None if pair is None else (pair.base.code, pair.quote.code)
    return QuoteKey.from_record(compact, currency=currency, fx_pair=pair)


def market_snapshot_to_records(snapshot: MarketDataSnapshot, /) -> FinanceRecordBatch:
    """Detach every quote vintage and all identity-bearing child records."""
    if not isinstance(snapshot, MarketDataSnapshot):
        raise TypeError("snapshot must be a MarketDataSnapshot.")
    rows = []
    for observation in snapshot.observations:
        rows.append(
            {
                "observation_id": observation.observation_id,
                "quote_key": _key_record(observation.key),
                "value": float(observation.value),
                "timestamp": _timestamp_record(observation.timestamp),
                "lineage": dict(observation.lineage.to_record()),
            }
        )
    return FinanceRecordBatch(
        "market-observation",
        rows,
        primary_key=("observation_id",),
        context={
            "snapshot_time": _timestamp_record(snapshot.snapshot_time),
            "reference_data_id": snapshot.reference_data_id,
            "snapshot_id": snapshot.snapshot_id,
        },
    )


def records_to_market_snapshot(batch: FinanceRecordBatch, /) -> MarketDataSnapshot:
    """Rebuild a typed market snapshot and verify every nested content address."""
    if not isinstance(batch, FinanceRecordBatch):
        raise TypeError("batch must be a FinanceRecordBatch.")
    if batch.record_kind != "market-observation":
        raise ValueError("Only market-observation records can form a market snapshot.")
    context = batch.context()
    if set(context) != {"snapshot_time", "reference_data_id", "snapshot_id"}:
        raise ValueError("Market-record context fields are not canonical.")
    snapshot_time_record = context["snapshot_time"]
    if not isinstance(snapshot_time_record, Mapping):
        raise TypeError("Serialized snapshot_time must be a mapping.")
    observations = []
    expected_row = {"observation_id", "quote_key", "value", "timestamp", "lineage"}
    for record in batch.to_records():
        if set(record) != expected_row:
            raise ValueError("Market observation record fields are not canonical.")
        key_record = record["quote_key"]
        timestamp_record = record["timestamp"]
        lineage_record = record["lineage"]
        if not isinstance(key_record, Mapping):
            raise TypeError("Serialized quote_key must be a mapping.")
        if not isinstance(timestamp_record, Mapping):
            raise TypeError("Serialized timestamp must be a mapping.")
        if not isinstance(lineage_record, Mapping):
            raise TypeError("Serialized lineage must be a mapping.")
        expected_lineage = {
            "source_id",
            "dataset_id",
            "publisher_id",
            "upstream_lineage_ids",
            "transformation_ids",
            "lineage_id",
        }
        if set(lineage_record) != expected_lineage:
            raise ValueError("Serialized lineage fields are not canonical.")
        key = _key_from_record(key_record)
        timestamp = _timestamp_from_record(timestamp_record)
        lineage = DataLineage.from_record(lineage_record)
        observation = QuoteObservation.from_record(
            record,
            key=key,
            timestamp=timestamp,
            lineage=lineage,
        )
        observations.append(observation)
    snapshot = MarketDataSnapshot.from_record(
        context,
        observations=observations,
        snapshot_time=_timestamp_from_record(snapshot_time_record),
    )
    return snapshot


__all__ = ["market_snapshot_to_records", "records_to_market_snapshot"]
