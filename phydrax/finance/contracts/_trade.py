#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..core import CurrencyAmount, FinancialTimestamp
from ._base import AbstractContract, AbstractResolvedContract
from ._exercise import SettlementTerms


def _quantity(value: ArrayLike, name: str, /, *, nonzero: bool) -> Array:
    host = np.asarray(value)
    if host.shape != () or host.dtype.kind not in "fiu" or not np.isfinite(host):
        raise ValueError(f"{name} must be one finite real scalar.")
    if nonzero and float(host) == 0.0:
        raise ValueError(f"{name} must be nonzero.")
    return jnp.asarray(host)


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


class Trade(StrictModule, NonTrainableState):
    """One executed transaction in a contract definition, not a current position."""

    trade_id: str = eqx.field(static=True)
    contract: AbstractContract
    quantity: Array
    execution_time: FinancialTimestamp = eqx.field(static=True)
    settlement: SettlementTerms
    transaction_price: CurrencyAmount | None
    trade_key: str = eqx.field(static=True)

    def __init__(
        self,
        trade_id: str,
        contract: AbstractContract,
        quantity: ArrayLike,
        execution_time: FinancialTimestamp,
        /,
        *,
        settlement: SettlementTerms,
        transaction_price: CurrencyAmount | None = None,
    ):
        identifier = _identifier(trade_id, "trade_id")
        if not isinstance(contract, AbstractContract):
            raise TypeError("contract must be an unresolved AbstractContract definition.")
        quantity_ = _quantity(quantity, "quantity", nonzero=True)
        if not isinstance(execution_time, FinancialTimestamp):
            raise TypeError("execution_time must be a FinancialTimestamp.")
        if not isinstance(settlement, SettlementTerms):
            raise TypeError("settlement must be SettlementTerms.")
        if transaction_price is not None and not isinstance(
            transaction_price, CurrencyAmount
        ):
            raise TypeError("transaction_price must be CurrencyAmount or None.")
        if (
            transaction_price is not None
            and transaction_price.currency.currency_id != settlement.currency.currency_id
        ):
            raise ValueError(
                "Transaction-price currency must match trade settlement currency."
            )
        self.trade_id = identifier
        self.contract = contract
        self.quantity = quantity_
        self.execution_time = execution_time
        self.settlement = settlement
        self.transaction_price = transaction_price
        self.trade_key = canonical_fingerprint(
            {
                "kind": "financial-trade",
                "trade_id": identifier,
                "contract": contract.contract_id,
                "quantity": float(np.asarray(quantity_)),
                "execution_event_ns": int(execution_time.epoch_nanoseconds),
                "execution_published_ns": int(execution_time.published_ns),
                "execution_received_ns": int(execution_time.received_ns),
                "execution_available_ns": int(execution_time.available_ns),
                "execution_vintage_id": execution_time.vintage_id,
                "settlement": settlement.terms_id,
                "transaction_price": None
                if transaction_price is None
                else {
                    "currency": transaction_price.currency.code,
                    "atoms": int(np.asarray(transaction_price.atoms)),
                },
            }
        )

    def to_record(self) -> Mapping[str, Any]:
        return {
            "trade_id": self.trade_id,
            "contract_id": self.contract.contract_id,
            "quantity": float(np.asarray(self.quantity)),
            "execution_time_ns": int(self.execution_time.epoch_nanoseconds),
            "settlement_terms_id": self.settlement.terms_id,
            "transaction_price": None
            if self.transaction_price is None
            else {
                "currency": self.transaction_price.currency.code,
                "atoms": int(np.asarray(self.transaction_price.atoms)),
            },
            "trade_key": self.trade_key,
        }


class Position(StrictModule, NonTrainableState):
    """Current quantity in a resolved contract, separate from execution evidence."""

    position_id: str = eqx.field(static=True)
    contract: AbstractResolvedContract
    quantity: Array
    account_id: str = eqx.field(static=True)
    position_key: str = eqx.field(static=True)

    def __init__(
        self,
        position_id: str,
        contract: AbstractResolvedContract,
        quantity: ArrayLike,
        /,
        *,
        account_id: str,
    ):
        identifier = _identifier(position_id, "position_id")
        account = _identifier(account_id, "account_id")
        if not isinstance(contract, AbstractResolvedContract):
            raise TypeError("contract must be a resolved AbstractResolvedContract.")
        quantity_ = _quantity(quantity, "quantity", nonzero=False)
        self.position_id = identifier
        self.contract = contract
        self.quantity = quantity_
        self.account_id = account
        self.position_key = canonical_fingerprint(
            {
                "kind": "financial-position",
                "position_id": identifier,
                "contract": contract.resolved_id,
                "quantity": float(np.asarray(quantity_)),
                "account_id": account,
            }
        )

    def to_record(self) -> Mapping[str, Any]:
        return {
            "position_id": self.position_id,
            "resolved_contract_id": self.contract.resolved_id,
            "quantity": float(np.asarray(self.quantity)),
            "account_id": self.account_id,
            "position_key": self.position_key,
        }


__all__ = ["Position", "Trade"]
