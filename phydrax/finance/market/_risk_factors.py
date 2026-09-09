#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._quotes import QuoteKey
from ._status import MarketStatus


class RiskFactorKey(StrictModule, NonTrainableState):
    """Named scalar coordinate backed by one exact market quote key."""

    name: str = eqx.field(static=True)
    quote_key: QuoteKey
    factor_id: str = eqx.field(static=True)

    def __init__(self, name: str, quote_key: QuoteKey, /):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("A risk-factor name must be a non-empty string.")
        if not isinstance(quote_key, QuoteKey):
            raise TypeError("quote_key must be a QuoteKey.")
        normalized = name.strip()
        self.name = normalized
        self.quote_key = quote_key
        self.factor_id = canonical_fingerprint(
            {
                "kind": "financial-risk-factor-key",
                "name": normalized,
                "quote_key": quote_key.key_id,
            }
        )

    def to_record(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "quote_key_id": self.quote_key.key_id,
            "factor_id": self.factor_id,
        }


class RiskFactorLayout(StrictModule, NonTrainableState):
    """Explicit ordered coordinate layout for device-side market arrays."""

    keys: tuple[RiskFactorKey, ...]
    factor_ids: tuple[str, ...] = eqx.field(static=True)
    quote_key_ids: tuple[str, ...] = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, keys: Sequence[RiskFactorKey], /):
        values = tuple(keys)
        if not values:
            raise ValueError("A risk-factor layout requires at least one key.")
        if not all(isinstance(value, RiskFactorKey) for value in values):
            raise TypeError("keys must contain only RiskFactorKey values.")
        identifiers = tuple(value.factor_id for value in values)
        quote_identifiers = tuple(value.quote_key.key_id for value in values)
        names = tuple(value.name for value in values)
        if (
            len(set(identifiers)) != len(identifiers)
            or len(set(quote_identifiers)) != len(quote_identifiers)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "Risk-factor layout factor, quote, and name identities must be unique."
            )
        self.keys = values
        self.factor_ids = identifiers
        self.quote_key_ids = quote_identifiers
        self.names = names
        self.layout_id = canonical_fingerprint(
            {
                "kind": "financial-risk-factor-layout",
                "factor_ids": list(identifiers),
            }
        )

    @property
    def factor_count(self) -> int:
        return len(self.keys)

    def index(self, key: RiskFactorKey | str, /) -> int:
        identifier = key.factor_id if isinstance(key, RiskFactorKey) else str(key)
        if identifier in self.factor_ids:
            return self.factor_ids.index(identifier)
        if identifier in self.names:
            return self.names.index(identifier)
        raise KeyError(f"Unknown risk factor {identifier!r}.")

    def to_record(self) -> Mapping[str, Any]:
        return {
            "keys": [dict(value.to_record()) for value in self.keys],
            "layout_id": self.layout_id,
        }


class MarketState(StrictModule, NonTrainableState):
    """Fixed-shape market coordinates with per-factor validity and status."""

    layout: RiskFactorLayout
    values: Array
    valid_mask: Array
    status: Array
    event_time_ns: Array
    availability_time_ns: Array
    decision_time_ns: int = eqx.field(static=True)
    observation_ids: tuple[str, ...] = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: RiskFactorLayout,
        values: ArrayLike,
        valid_mask: ArrayLike,
        status: ArrayLike,
        event_time_ns: ArrayLike,
        availability_time_ns: ArrayLike,
        /,
        *,
        decision_time_ns: int,
        observation_ids: Sequence[str] = (),
    ):
        if not isinstance(layout, RiskFactorLayout):
            raise TypeError("layout must be a RiskFactorLayout.")
        count = layout.factor_count
        values_host = np.asarray(values)
        valid_host = np.asarray(valid_mask, dtype=bool)
        status_host = np.asarray(status, dtype=np.int32)
        event_host = np.asarray(event_time_ns, dtype=np.int64)
        available_host = np.asarray(availability_time_ns, dtype=np.int64)
        if values_host.shape != (count,) or values_host.dtype.kind not in "fiu":
            raise ValueError("values must be one real scalar per risk factor.")
        if valid_host.shape != (count,):
            raise ValueError("valid_mask must have shape (factor_count,).")
        if status_host.shape != (count,):
            raise ValueError("status must have shape (factor_count,).")
        if event_host.shape != (count,) or available_host.shape != (count,):
            raise ValueError("market-state time arrays must have shape (factor_count,).")
        if np.any(~np.isfinite(values_host)):
            raise ValueError("Market-state values must use finite numeric storage.")
        if not np.array_equal(valid_host, status_host == int(MarketStatus.SUCCESS)):
            raise ValueError(
                "valid_mask must exactly identify factors with successful status."
            )
        if isinstance(decision_time_ns, bool) or not isinstance(decision_time_ns, int):
            raise TypeError("decision_time_ns must be an integer.")
        identifiers = tuple(str(value) for value in observation_ids)
        if not identifiers:
            identifiers = ("",) * count
        if len(identifiers) != count:
            raise ValueError("observation_ids must contain one entry per factor.")
        causal = (~valid_host) | (available_host <= decision_time_ns)
        if not np.all(causal):
            raise ValueError(
                "A valid market factor cannot be available after the decision."
            )
        valid_host = valid_host.copy()
        self.layout = layout
        self.values = jnp.asarray(values_host)
        self.valid_mask = jnp.asarray(valid_host)
        self.status = jnp.asarray(status_host)
        self.event_time_ns = jnp.asarray(event_host)
        self.availability_time_ns = jnp.asarray(available_host)
        self.decision_time_ns = decision_time_ns
        self.observation_ids = identifiers
        self.state_id = canonical_fingerprint(
            {
                "kind": "financial-market-state",
                "layout": layout.layout_id,
                "values": values_host.tolist(),
                "valid_mask": valid_host.tolist(),
                "status": status_host.tolist(),
                "event_time_ns": event_host.tolist(),
                "availability_time_ns": available_host.tolist(),
                "decision_time_ns": decision_time_ns,
                "observation_ids": list(identifiers),
            }
        )

    @property
    def accepted(self) -> Array:
        return self.valid_mask & (self.status == int(MarketStatus.SUCCESS))

    @property
    def successful(self) -> Array:
        return jnp.all(self.accepted)

    def value(self, key: RiskFactorKey | str, /) -> Array:
        index = self.layout.index(key)
        if not bool(np.asarray(self.accepted[index])):
            raise ValueError(f"Risk factor {self.layout.names[index]!r} is not accepted.")
        return self.values[index]

    def reorder(self, layout: RiskFactorLayout, /) -> MarketState:
        """Reindex by semantic factor identity; never rely on array position."""

        if not isinstance(layout, RiskFactorLayout):
            raise TypeError("layout must be a RiskFactorLayout.")
        source = {
            identifier: index for index, identifier in enumerate(self.layout.factor_ids)
        }
        source_values = np.asarray(self.values)
        source_valid = np.asarray(self.valid_mask)
        source_status = np.asarray(self.status)
        source_event = np.asarray(self.event_time_ns)
        source_available = np.asarray(self.availability_time_ns)
        count = layout.factor_count
        values = np.zeros((count,), dtype=source_values.dtype)
        valid = np.zeros((count,), dtype=bool)
        status = np.full((count,), int(MarketStatus.MISSING_FACTOR), dtype=np.int32)
        event = np.zeros((count,), dtype=np.int64)
        available = np.zeros((count,), dtype=np.int64)
        observations = [""] * count
        for target, identifier in enumerate(layout.factor_ids):
            if identifier not in source:
                continue
            index = source[identifier]
            values[target] = source_values[index]
            valid[target] = source_valid[index]
            status[target] = source_status[index]
            event[target] = source_event[index]
            available[target] = source_available[index]
            observations[target] = self.observation_ids[index]
        return MarketState(
            layout,
            values,
            valid,
            status,
            event,
            available,
            decision_time_ns=self.decision_time_ns,
            observation_ids=tuple(observations),
        )

    def to_record(self) -> Mapping[str, Any]:
        return {
            "layout_id": self.layout.layout_id,
            "values": np.asarray(self.values).tolist(),
            "valid_mask": np.asarray(self.valid_mask).tolist(),
            "status": np.asarray(self.status).tolist(),
            "event_time_ns": np.asarray(self.event_time_ns).tolist(),
            "availability_time_ns": np.asarray(self.availability_time_ns).tolist(),
            "decision_time_ns": self.decision_time_ns,
            "observation_ids": list(self.observation_ids),
            "state_id": self.state_id,
        }


__all__ = ["MarketState", "RiskFactorKey", "RiskFactorLayout"]
