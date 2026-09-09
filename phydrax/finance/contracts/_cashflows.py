#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntFlag
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..core import Currency, FinanceDate


class CashflowStatus(IntFlag):
    """Fail-closed preparation status for known cashflow obligations."""

    SUCCESS = 0
    CAPACITY_EXCEEDED = 1


class CashflowBatch(StrictModule, NonTrainableState):
    """Fixed-shape known obligations; unresolved floating formulas do not belong here."""

    payment_ordinals: Array
    amounts: Array
    currencies: tuple[Currency, ...]
    currency_index: Array
    valid_mask: Array
    status: Array
    obligation_ids: tuple[str, ...] = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)

    def __init__(
        self,
        payment_dates: Sequence[FinanceDate],
        amounts: ArrayLike,
        currencies: Sequence[Currency],
        /,
        *,
        obligation_ids: Sequence[str] = (),
        capacity: int | None = None,
    ):
        dates = tuple(payment_dates)
        currency_values = tuple(currencies)
        amount_host = np.asarray(amounts)
        if not all(isinstance(value, FinanceDate) for value in dates):
            raise TypeError("payment_dates must contain only FinanceDate values.")
        if not all(isinstance(value, Currency) for value in currency_values):
            raise TypeError("currencies must contain only Currency values.")
        if amount_host.ndim != 1 or amount_host.dtype.kind not in "fiu":
            raise ValueError("amounts must be a rank-one real array.")
        count = len(dates)
        if amount_host.shape != (count,) or len(currency_values) != count:
            raise ValueError(
                "Cashflow dates, amounts, and currencies must have equal length."
            )
        if np.any(~np.isfinite(amount_host)):
            raise ValueError("Known cashflow amounts must be finite.")
        if capacity is None:
            capacity_ = count
        elif (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity < count
        ):
            raise ValueError(
                "capacity must be an integer at least as large as the cashflow count."
            )
        else:
            capacity_ = capacity
        identifiers = tuple(str(value).strip() for value in obligation_ids)
        if not identifiers:
            identifiers = tuple(
                canonical_fingerprint(
                    {
                        "kind": "known-cashflow-obligation",
                        "ordinal": date.ordinal,
                        "amount": float(amount),
                        "currency_id": currency.currency_id,
                        "position": index,
                    }
                )
                for index, (date, amount, currency) in enumerate(
                    zip(dates, amount_host, currency_values, strict=True)
                )
            )
        if len(identifiers) != count or any(not value for value in identifiers):
            raise ValueError("obligation_ids must contain one non-empty ID per cashflow.")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Known cashflow obligation IDs must be unique.")
        universe_by_id: dict[str, Currency] = {}
        for currency in currency_values:
            universe_by_id.setdefault(currency.currency_id, currency)
        universe = tuple(
            universe_by_id[identifier] for identifier in sorted(universe_by_id)
        )
        universe_index = {
            currency.currency_id: index for index, currency in enumerate(universe)
        }
        payment = np.zeros((capacity_,), dtype=np.int32)
        amount_values = np.zeros((capacity_,), dtype=amount_host.dtype)
        currency_indices = np.zeros((capacity_,), dtype=np.int32)
        valid = np.zeros((capacity_,), dtype=bool)
        status = np.zeros((capacity_,), dtype=np.int32)
        if count:
            payment[:count] = np.asarray(
                [value.ordinal for value in dates], dtype=np.int32
            )
            amount_values[:count] = amount_host
            currency_indices[:count] = np.asarray(
                [universe_index[value.currency_id] for value in currency_values],
                dtype=np.int32,
            )
            valid[:count] = True
        padded_ids = identifiers + ("",) * (capacity_ - count)
        self.payment_ordinals = jnp.asarray(payment)
        self.amounts = jnp.asarray(amount_values)
        self.currencies = universe
        self.currency_index = jnp.asarray(currency_indices)
        self.valid_mask = jnp.asarray(valid)
        self.status = jnp.asarray(status)
        self.obligation_ids = padded_ids
        self.capacity = capacity_
        self.batch_id = canonical_fingerprint(
            {
                "kind": "known-financial-cashflow-batch",
                "payment_ordinals": payment.tolist(),
                "amounts": amount_values.tolist(),
                "currency_ids": [value.currency_id for value in universe],
                "currency_index": currency_indices.tolist(),
                "valid_mask": valid.tolist(),
                "status": status.tolist(),
                "obligation_ids": list(padded_ids),
            }
        )

    @property
    def active_count(self) -> int:
        return int(np.sum(np.asarray(self.valid_mask)))

    @property
    def accepted(self) -> Array:
        return self.valid_mask & (self.status == int(CashflowStatus.SUCCESS))

    def currency_for(self, index: int, /) -> Currency:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer.")
        if (
            index < 0
            or index >= self.capacity
            or not bool(np.asarray(self.valid_mask[index]))
        ):
            raise IndexError("index does not identify an active cashflow.")
        return self.currencies[int(np.asarray(self.currency_index[index]))]

    def pad(self, capacity: int, /) -> CashflowBatch:
        """Return the same obligations with neutral zero padding to ``capacity``."""

        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity < self.active_count
        ):
            raise ValueError("capacity must not truncate known obligations.")
        dates = tuple(
            FinanceDate(int(value))
            for value in np.asarray(self.payment_ordinals)[np.asarray(self.valid_mask)]
        )
        amounts = np.asarray(self.amounts)[np.asarray(self.valid_mask)]
        currencies = tuple(
            self.currencies[int(value)]
            for value in np.asarray(self.currency_index)[np.asarray(self.valid_mask)]
        )
        identifiers = tuple(
            value
            for value, active in zip(
                self.obligation_ids, np.asarray(self.valid_mask), strict=True
            )
            if active
        )
        return CashflowBatch(
            dates,
            amounts,
            currencies,
            obligation_ids=identifiers,
            capacity=capacity,
        )

    def prepare(self, capacity: int, /) -> PreparedCashflowBatch:
        """Prepare a fixed-capacity device batch and report truncation without hiding it."""

        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer.")
        if self.active_count <= capacity:
            return PreparedCashflowBatch(
                self.pad(capacity),
                jnp.asarray(int(CashflowStatus.SUCCESS), dtype=jnp.int32),
            )
        host_valid = np.asarray(self.valid_mask)
        active = np.flatnonzero(host_valid)[:capacity]
        dates = tuple(
            FinanceDate(int(np.asarray(self.payment_ordinals)[index])) for index in active
        )
        amounts = np.asarray(self.amounts)[active]
        currencies = tuple(
            self.currencies[int(np.asarray(self.currency_index)[index])]
            for index in active
        )
        identifiers = tuple(self.obligation_ids[index] for index in active)
        truncated = CashflowBatch(
            dates, amounts, currencies, obligation_ids=identifiers, capacity=capacity
        )
        failed = PreparedCashflowBatch(
            truncated,
            jnp.asarray(int(CashflowStatus.CAPACITY_EXCEEDED), dtype=jnp.int32),
        )
        return failed

    def to_record(self) -> Mapping[str, Any]:
        return {
            "payment_ordinals": np.asarray(self.payment_ordinals).tolist(),
            "amounts": np.asarray(self.amounts).tolist(),
            "currencies": [value.code for value in self.currencies],
            "currency_index": np.asarray(self.currency_index).tolist(),
            "valid_mask": np.asarray(self.valid_mask).tolist(),
            "status": np.asarray(self.status).tolist(),
            "obligation_ids": list(self.obligation_ids),
            "batch_id": self.batch_id,
        }


class PreparedCashflowBatch(StrictModule, NonTrainableState):
    """Fixed-capacity known cashflows plus a global fail-closed preparation status."""

    cashflows: CashflowBatch
    preparation_status: Array

    @property
    def accepted(self) -> Array:
        return self.cashflows.accepted & (
            self.preparation_status == int(CashflowStatus.SUCCESS)
        )


__all__ = ["CashflowBatch", "CashflowStatus", "PreparedCashflowBatch"]
