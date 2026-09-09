#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..core import Currency, FinancialTimestamp, FXPair
from ._lineage import DataLineage
from ._status import MarketStatus


class QuoteTiePolicy(str, Enum):
    """Explicit resolution of indistinguishable bitemporal quote coordinates."""

    REJECT = "reject"
    EARLIEST_VINTAGE = "earliest_vintage"
    LATEST_VINTAGE = "latest_vintage"


def _nonempty(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _optional_text(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return value.strip()


def _event_ns(value: FinancialTimestamp, name: str, /) -> int:
    if not isinstance(value, FinancialTimestamp):
        raise TypeError(f"{name} must be a FinancialTimestamp.")
    return int(value.epoch_nanoseconds)


def _currency_code(value: Currency | None, /) -> str | None:
    return None if value is None else value.code


def _pair_codes(value: FXPair | None, /) -> tuple[str, str] | None:
    return None if value is None else (value.base.code, value.quote.code)


class QuoteKey(StrictModule, NonTrainableState):
    """Stable identity of one quoted field, independent of its observations."""

    reference_id: str = eqx.field(static=True)
    field: str = eqx.field(static=True)
    venue: str = eqx.field(static=True)
    currency: Currency | None = eqx.field(static=True)
    fx_pair: FXPair | None = eqx.field(static=True)
    key_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_id: str,
        field: str,
        /,
        *,
        venue: str = "",
        currency: Currency | None = None,
        fx_pair: FXPair | None = None,
    ):
        reference = _nonempty(reference_id, "reference_id")
        field_ = _nonempty(field, "field").lower()
        venue_ = _optional_text(venue, "venue")
        if currency is not None and not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency or None.")
        if fx_pair is not None and not isinstance(fx_pair, FXPair):
            raise TypeError("fx_pair must be an FXPair or None.")
        if currency is not None and fx_pair is not None:
            raise ValueError("A quote key cannot declare both currency and fx_pair.")
        self.reference_id = reference
        self.field = field_
        self.venue = venue_
        self.currency = currency
        self.fx_pair = fx_pair
        self.key_id = canonical_fingerprint(
            {
                "kind": "financial-quote-key",
                "reference_id": reference,
                "field": field_,
                "venue": venue_,
                "currency_id": None if currency is None else currency.currency_id,
                "fx_pair_id": None if fx_pair is None else fx_pair.pair_id,
            }
        )

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        /,
        *,
        currency: Currency | None = None,
        fx_pair: FXPair | None = None,
    ) -> QuoteKey:
        value = cls(
            str(record["reference_id"]),
            str(record["field"]),
            venue=str(record["venue"]),
            currency=currency,
            fx_pair=fx_pair,
        )
        if "key_id" in record and str(record["key_id"]) != value.key_id:
            raise ValueError("Quote-key record content does not match key_id.")
        return value

    def to_record(self) -> Mapping[str, Any]:
        return {
            "reference_id": self.reference_id,
            "field": self.field,
            "venue": self.venue,
            "currency": _currency_code(self.currency),
            "fx_pair": _pair_codes(self.fx_pair),
            "key_id": self.key_id,
        }


class QuoteObservation(StrictModule, NonTrainableState):
    """One immutable quote vintage with effective and availability clocks."""

    key: QuoteKey
    value: Array
    timestamp: FinancialTimestamp = eqx.field(static=True)
    lineage: DataLineage
    observation_id: str = eqx.field(static=True)

    def __init__(
        self,
        key: QuoteKey,
        value: ArrayLike,
        timestamp: FinancialTimestamp,
        lineage: DataLineage,
        /,
    ):
        if not isinstance(key, QuoteKey):
            raise TypeError("key must be a QuoteKey.")
        scalar = np.asarray(value)
        if scalar.shape != () or scalar.dtype.kind not in "fiu":
            raise ValueError("A quote observation value must be one real scalar.")
        if not isinstance(timestamp, FinancialTimestamp):
            raise TypeError("timestamp must be a FinancialTimestamp.")
        if not isinstance(lineage, DataLineage):
            raise TypeError("lineage must be a DataLineage.")
        self.key = key
        self.value = jnp.asarray(scalar)
        self.timestamp = timestamp
        self.lineage = lineage
        self.observation_id = canonical_fingerprint(
            {
                "kind": "financial-quote-observation",
                "key": key.key_id,
                "value": float(scalar),
                "event_time_ns": int(timestamp.epoch_nanoseconds),
                "published_time_ns": int(timestamp.published_ns),
                "received_time_ns": int(timestamp.received_ns),
                "available_time_ns": int(timestamp.available_ns),
                "vintage_id": timestamp.vintage_id,
                "lineage": lineage.lineage_id,
            }
        )

    @property
    def event_time_ns(self) -> int:
        return int(self.timestamp.epoch_nanoseconds)

    @property
    def available_time_ns(self) -> int:
        return int(self.timestamp.available_ns)

    @property
    def vintage_id(self) -> str:
        return self.timestamp.vintage_id

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        /,
        *,
        key: QuoteKey,
        timestamp: FinancialTimestamp,
        lineage: DataLineage,
    ) -> QuoteObservation:
        value = cls(key, record["value"], timestamp, lineage)
        if (
            "observation_id" in record
            and str(record["observation_id"]) != value.observation_id
        ):
            raise ValueError(
                "Quote-observation record content does not match observation_id."
            )
        return value

    def to_record(self) -> Mapping[str, Any]:
        return {
            "key_id": self.key.key_id,
            "value": float(np.asarray(self.value)),
            "event_time_ns": self.event_time_ns,
            "published_time_ns": int(self.timestamp.published_ns),
            "received_time_ns": int(self.timestamp.received_ns),
            "available_time_ns": self.available_time_ns,
            "vintage_id": self.vintage_id,
            "lineage_id": self.lineage.lineage_id,
            "observation_id": self.observation_id,
        }


class QuoteSelection(StrictModule):
    """Audited scalar result of a host-side bitemporal selection."""

    value: Array
    valid: Array
    status: Array
    event_time_ns: Array
    available_time_ns: Array
    observation_id: str = eqx.field(static=True)
    duplicate_count: int = eqx.field(static=True)


class FixingSeries(StrictModule, NonTrainableState):
    """All vintages of one fixing key, with causal point-in-time lookup."""

    key: QuoteKey
    observations: tuple[QuoteObservation, ...]
    series_id: str = eqx.field(static=True)

    def __init__(self, key: QuoteKey, observations: Sequence[QuoteObservation], /):
        if not isinstance(key, QuoteKey):
            raise TypeError("key must be a QuoteKey.")
        values = tuple(observations)
        if not values:
            raise ValueError("A fixing series requires at least one observation.")
        if not all(isinstance(value, QuoteObservation) for value in values):
            raise TypeError("observations must contain QuoteObservation values.")
        if any(value.key.key_id != key.key_id for value in values):
            raise ValueError("Every fixing observation must have the series quote key.")
        ordered = tuple(
            sorted(
                values,
                key=lambda value: (
                    value.event_time_ns,
                    value.available_time_ns,
                    value.vintage_id,
                    value.observation_id,
                ),
            )
        )
        if len({value.observation_id for value in ordered}) != len(ordered):
            raise ValueError("A fixing series cannot contain the same vintage twice.")
        self.key = key
        self.observations = ordered
        self.series_id = canonical_fingerprint(
            {
                "kind": "financial-fixing-series",
                "key": key.key_id,
                "observations": [value.observation_id for value in ordered],
            }
        )

    @property
    def vintage_count(self) -> int:
        return len(self.observations)

    def at(
        self,
        event_time: FinancialTimestamp,
        decision_time: FinancialTimestamp,
        /,
        *,
        tie_policy: QuoteTiePolicy = QuoteTiePolicy.REJECT,
    ) -> QuoteSelection:
        event_ns = _event_ns(event_time, "event_time")
        decision_ns = _event_ns(decision_time, "decision_time")
        if not isinstance(tie_policy, QuoteTiePolicy):
            raise TypeError("tie_policy must be a QuoteTiePolicy.")
        event_matches = [
            value for value in self.observations if value.event_time_ns == event_ns
        ]
        candidates = [
            value for value in event_matches if value.available_time_ns <= decision_ns
        ]
        if not candidates:
            status = (
                MarketStatus.CAUSAL_TIME_VIOLATION
                if event_matches
                else MarketStatus.MISSING_FACTOR
            )
            return _missing_selection(status)
        newest_availability = max(value.available_time_ns for value in candidates)
        newest = [
            value
            for value in candidates
            if value.available_time_ns == newest_availability
        ]
        duplicates = len(newest)
        if duplicates > 1 and tie_policy is QuoteTiePolicy.REJECT:
            return _missing_selection(MarketStatus.DUPLICATE, duplicates=duplicates)
        if tie_policy is QuoteTiePolicy.EARLIEST_VINTAGE:
            selected = min(
                newest, key=lambda value: (value.vintage_id, value.observation_id)
            )
        else:
            selected = max(
                newest, key=lambda value: (value.vintage_id, value.observation_id)
            )
        finite = bool(np.isfinite(np.asarray(selected.value)))
        status = MarketStatus.SUCCESS if finite else MarketStatus.NONFINITE
        return QuoteSelection(
            selected.value,
            jnp.asarray(finite),
            jnp.asarray(int(status), dtype=jnp.int32),
            jnp.asarray(selected.event_time_ns, dtype=jnp.int64),
            jnp.asarray(selected.available_time_ns, dtype=jnp.int64),
            selected.observation_id,
            duplicates,
        )


def _missing_selection(status: MarketStatus, /, *, duplicates: int = 0) -> QuoteSelection:
    return QuoteSelection(
        jnp.asarray(0.0),
        jnp.asarray(False),
        jnp.asarray(int(status), dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int64),
        jnp.asarray(0, dtype=jnp.int64),
        "",
        duplicates,
    )


__all__ = [
    "FixingSeries",
    "QuoteKey",
    "QuoteObservation",
    "QuoteSelection",
    "QuoteTiePolicy",
]
