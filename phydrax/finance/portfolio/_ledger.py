#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..core import (
    add_currency_amounts,
    AssetReference,
    Currency,
    CurrencyAmount,
    subtract_currency_amounts,
)


def _scalar(value: ArrayLike, name: str, /, *, nonzero: bool = False) -> Array:
    result = jnp.asarray(value)
    if result.shape != () or jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be one real scalar.")
    result = result.astype(jnp.result_type(result.dtype, jnp.float32))
    raw = float(np.asarray(result))
    if not np.isfinite(raw) or (nonzero and raw == 0.0):
        raise ValueError(f"{name} must be finite{' and nonzero' if nonzero else ''}.")
    return result


def _id(value: str, name: str, /) -> str:
    result = str(value)
    if not result or result != result.strip():
        raise ValueError(f"{name} must be non-empty without surrounding whitespace.")
    return result


def _same_currency(*amounts: CurrencyAmount) -> Currency:
    if not amounts or any(not isinstance(amount, CurrencyAmount) for amount in amounts):
        raise TypeError("Exact ledger amounts must be CurrencyAmount values.")
    currency = amounts[0].currency
    if any(amount.currency.currency_id != currency.currency_id for amount in amounts[1:]):
        raise ValueError("A ledger posting cannot mix exact currencies.")
    return currency


class Holding(StrictModule):
    """Immutable quantity held in one referenced asset."""

    asset: AssetReference = eqx.field(static=True)
    quantity: Array

    def __init__(self, asset: AssetReference, quantity: ArrayLike, /):
        if not isinstance(asset, AssetReference):
            raise TypeError("asset must be an AssetReference.")
        self.asset = asset
        self.quantity = _scalar(quantity, "quantity")


class CashBalance(StrictModule):
    """Exact cash balance in a named account and currency."""

    amount: CurrencyAmount
    account_id: str = eqx.field(static=True)

    def __init__(self, account_id: str, amount: CurrencyAmount, /):
        if not isinstance(amount, CurrencyAmount):
            raise TypeError("amount must be a CurrencyAmount.")
        self.account_id = _id(account_id, "account_id")
        self.amount = amount

    @property
    def currency(self) -> Currency:
        return self.amount.currency


class TaxLot(StrictModule):
    """Open quantity and exact aggregate basis for one acquisition lot."""

    asset: AssetReference = eqx.field(static=True)
    quantity: Array
    cost_basis: CurrencyAmount
    lot_id: str = eqx.field(static=True)
    acquired_index: int = eqx.field(static=True)

    def __init__(
        self,
        lot_id: str,
        asset: AssetReference,
        quantity: ArrayLike,
        cost_basis: CurrencyAmount,
        /,
        *,
        acquired_index: int,
    ):
        if not isinstance(asset, AssetReference):
            raise TypeError("asset must be an AssetReference.")
        if not isinstance(cost_basis, CurrencyAmount):
            raise TypeError("cost_basis must be a CurrencyAmount.")
        if cost_basis.currency.currency_id != asset.currency.currency_id:
            raise ValueError("Tax-lot basis must use the asset currency.")
        quantity_ = _scalar(quantity, "quantity")
        basis_atoms = int(np.asarray(cost_basis.atoms))
        if (
            float(np.asarray(quantity_)) < 0.0
            or basis_atoms < 0
            or ((float(np.asarray(quantity_)) == 0.0) != (basis_atoms == 0))
        ):
            raise ValueError(
                "Tax-lot quantity and basis must be non-negative and vanish together."
            )
        acquired = int(acquired_index)
        if acquired < 0:
            raise ValueError("acquired_index must be non-negative.")
        self.lot_id, self.asset = _id(lot_id, "lot_id"), asset
        self.quantity, self.cost_basis = quantity_, cost_basis
        self.acquired_index = acquired


class LedgerTrade(StrictModule):
    """Booked trade evidence with exact principal, fee, and tax postings."""

    asset: AssetReference = eqx.field(static=True)
    quantity: Array
    principal_cash_flow: CurrencyAmount
    fees: CurrencyAmount
    tax_cost: CurrencyAmount
    trade_id: str = eqx.field(static=True)
    account_id: str = eqx.field(static=True)
    execution_index: int = eqx.field(static=True)
    settlement_index: int = eqx.field(static=True)

    def __init__(
        self,
        trade_id: str,
        account_id: str,
        asset: AssetReference,
        quantity: ArrayLike,
        principal_cash_flow: CurrencyAmount,
        fees: CurrencyAmount,
        tax_cost: CurrencyAmount,
        /,
        *,
        execution_index: int,
        settlement_index: int,
    ):
        if not isinstance(asset, AssetReference):
            raise TypeError("asset must be an AssetReference.")
        currency = _same_currency(principal_cash_flow, fees, tax_cost)
        if currency.currency_id != asset.currency.currency_id:
            raise ValueError("Trade cash postings must use the asset currency.")
        quantity_ = _scalar(quantity, "quantity", nonzero=True)
        if int(np.asarray(fees.atoms)) < 0 or int(np.asarray(tax_cost.atoms)) < 0:
            raise ValueError("Trade fees and tax costs must be non-negative.")
        if float(np.asarray(quantity_)) * int(np.asarray(principal_cash_flow.atoms)) >= 0:
            raise ValueError(
                "Principal cash flow must be nonzero and opposite the trade quantity."
            )
        execution, settlement = int(execution_index), int(settlement_index)
        if execution < 0 or settlement < execution:
            raise ValueError("Trade indices require 0 <= execution <= settlement.")
        self.trade_id, self.account_id = (
            _id(trade_id, "trade_id"),
            _id(account_id, "account_id"),
        )
        self.asset = asset
        self.quantity = quantity_
        self.principal_cash_flow = principal_cash_flow
        self.fees, self.tax_cost = fees, tax_cost
        self.execution_index, self.settlement_index = execution, settlement


class LotLedgerEntry(StrictModule):
    """Explicit tax-lot quantity and exact basis movement."""

    asset: AssetReference = eqx.field(static=True)
    quantity_delta: Array
    basis_delta: CurrencyAmount
    entry_id: str = eqx.field(static=True)
    lot_id: str = eqx.field(static=True)
    effective_index: int = eqx.field(static=True)

    def __init__(
        self,
        entry_id: str,
        lot_id: str,
        asset: AssetReference,
        quantity_delta: ArrayLike,
        basis_delta: CurrencyAmount,
        /,
        *,
        effective_index: int,
    ):
        if not isinstance(asset, AssetReference):
            raise TypeError("asset must be an AssetReference.")
        if not isinstance(basis_delta, CurrencyAmount):
            raise TypeError("basis_delta must be a CurrencyAmount.")
        if basis_delta.currency.currency_id != asset.currency.currency_id:
            raise ValueError("Lot basis movement must use the asset currency.")
        quantity_ = _scalar(quantity_delta, "quantity_delta", nonzero=True)
        if float(np.asarray(quantity_)) * int(np.asarray(basis_delta.atoms)) < 0:
            raise ValueError(
                "Lot quantity and basis movements cannot have opposite signs."
            )
        effective = int(effective_index)
        if effective < 0:
            raise ValueError("effective_index must be non-negative.")
        self.entry_id, self.lot_id = _id(entry_id, "entry_id"), _id(lot_id, "lot_id")
        self.asset = asset
        self.quantity_delta = quantity_
        self.basis_delta = basis_delta
        self.effective_index = effective


class PortfolioLedger(StrictModule):
    """Append-only immutable holdings, cash, trade, and tax-lot ledger."""

    initial_holdings: tuple[Holding, ...]
    initial_cash: tuple[CashBalance, ...]
    initial_lots: tuple[TaxLot, ...]
    trades: tuple[LedgerTrade, ...]
    lot_entries: tuple[LotLedgerEntry, ...]
    base_currency: Currency = eqx.field(static=True)
    ledger_id: str = eqx.field(static=True)

    def __init__(
        self,
        ledger_id: str,
        base_currency: Currency,
        /,
        *,
        initial_holdings: tuple[Holding, ...] = (),
        initial_cash: tuple[CashBalance, ...] = (),
        initial_lots: tuple[TaxLot, ...] = (),
        trades: tuple[LedgerTrade, ...] = (),
        lot_entries: tuple[LotLedgerEntry, ...] = (),
    ):
        if not isinstance(base_currency, Currency):
            raise TypeError("base_currency must be a Currency.")
        holdings, cash, lots = (
            tuple(initial_holdings),
            tuple(initial_cash),
            tuple(initial_lots),
        )
        trades_, entries = tuple(trades), tuple(lot_entries)
        expected = (
            (holdings, Holding, "initial_holdings"),
            (cash, CashBalance, "initial_cash"),
            (lots, TaxLot, "initial_lots"),
            (trades_, LedgerTrade, "trades"),
            (entries, LotLedgerEntry, "lot_entries"),
        )
        for values, cls, name in expected:
            if any(not isinstance(value, cls) for value in values):
                raise TypeError(f"{name} contains an invalid value.")
        keys = (
            tuple(item.asset.asset_id for item in holdings),
            tuple((item.account_id, item.currency.currency_id) for item in cash),
            tuple(item.lot_id for item in lots),
            tuple(item.trade_id for item in trades_),
            tuple(item.entry_id for item in entries),
        )
        if any(len(set(group)) != len(group) for group in keys):
            raise ValueError("Initial ledger keys and event identifiers must be unique.")
        if (
            tuple(sorted(trades_, key=lambda item: (item.execution_index, item.trade_id)))
            != trades_
        ):
            raise ValueError("trades must be ordered by (execution_index, trade_id).")
        if (
            tuple(sorted(entries, key=lambda item: (item.effective_index, item.entry_id)))
            != entries
        ):
            raise ValueError(
                "lot_entries must be ordered by (effective_index, entry_id)."
            )
        self.ledger_id, self.base_currency = _id(ledger_id, "ledger_id"), base_currency
        self.initial_holdings, self.initial_cash, self.initial_lots = holdings, cash, lots
        self.trades, self.lot_entries = trades_, entries


class PortfolioLedgerSnapshot(StrictModule):
    holdings: tuple[Holding, ...]
    cash: tuple[CashBalance, ...]
    lots: tuple[TaxLot, ...]
    as_of_index: int = eqx.field(static=True)
    ledger_id: str = eqx.field(static=True)


def portfolio_ledger_snapshot(
    ledger: PortfolioLedger,
    as_of_index: int,
    /,
) -> PortfolioLedgerSnapshot:
    """Replay ledger events only; no optimizer or compiled matrix is consulted."""

    if not isinstance(ledger, PortfolioLedger):
        raise TypeError("ledger must be a PortfolioLedger.")
    index = int(as_of_index)
    if index < 0:
        raise ValueError("as_of_index must be non-negative.")
    asset_by_id = {item.asset.asset_id: item.asset for item in ledger.initial_holdings}
    quantity = {item.asset.asset_id: item.quantity for item in ledger.initial_holdings}
    for trade in ledger.trades:
        asset_by_id[trade.asset.asset_id] = trade.asset
        if trade.execution_index <= index:
            prior = quantity.get(
                trade.asset.asset_id, jnp.asarray(0.0, dtype=trade.quantity.dtype)
            )
            quantity[trade.asset.asset_id] = prior + trade.quantity
    holdings = tuple(
        Holding(asset_by_id[asset_id], quantity[asset_id])
        for asset_id in sorted(quantity)
    )
    cash_key = lambda item: (item.account_id, item.currency.currency_id)
    amounts = {cash_key(item): item.amount for item in ledger.initial_cash}
    for trade in ledger.trades:
        if trade.settlement_index <= index:
            key = (trade.account_id, trade.asset.currency.currency_id)
            previous = amounts.get(
                key,
                CurrencyAmount(
                    trade.asset.currency,
                    jnp.asarray(0, dtype=trade.principal_cash_flow.atoms.dtype),
                ),
            )
            with_principal = add_currency_amounts(
                previous,
                trade.principal_cash_flow,
            )
            after_fees = subtract_currency_amounts(with_principal, trade.fees)
            amounts[key] = subtract_currency_amounts(after_fees, trade.tax_cost)
    cash = tuple(
        CashBalance(account, amounts[(account, currency_id)])
        for account, currency_id in sorted(amounts)
    )
    lot_asset = {item.lot_id: item.asset for item in ledger.initial_lots}
    lot_acquired = {item.lot_id: item.acquired_index for item in ledger.initial_lots}
    lot_quantity = {item.lot_id: item.quantity for item in ledger.initial_lots}
    lot_basis = {item.lot_id: item.cost_basis for item in ledger.initial_lots}
    for entry in ledger.lot_entries:
        if entry.effective_index <= index:
            lot_asset[entry.lot_id] = entry.asset
            lot_acquired.setdefault(entry.lot_id, entry.effective_index)
            lot_quantity[entry.lot_id] = (
                lot_quantity.get(
                    entry.lot_id, jnp.asarray(0.0, dtype=entry.quantity_delta.dtype)
                )
                + entry.quantity_delta
            )
            previous_basis = lot_basis.get(
                entry.lot_id,
                CurrencyAmount(
                    entry.asset.currency,
                    jnp.asarray(0, dtype=entry.basis_delta.atoms.dtype),
                ),
            )
            lot_basis[entry.lot_id] = add_currency_amounts(
                previous_basis,
                entry.basis_delta,
            )
    for lot_id in lot_quantity:
        quantity_value = float(np.asarray(lot_quantity[lot_id]))
        basis_value = int(np.asarray(lot_basis[lot_id].atoms))
        if (
            quantity_value < 0.0
            or basis_value < 0
            or ((quantity_value == 0.0) != (basis_value == 0))
        ):
            raise ValueError(
                f"Lot {lot_id!r} replayed to an invalid quantity/basis state."
            )
    lots = tuple(
        TaxLot(
            lot_id,
            lot_asset[lot_id],
            lot_quantity[lot_id],
            lot_basis[lot_id],
            acquired_index=lot_acquired[lot_id],
        )
        for lot_id in sorted(lot_quantity)
        if float(np.asarray(lot_quantity[lot_id])) > 0.0
    )
    return PortfolioLedgerSnapshot(
        holdings=holdings,
        cash=cash,
        lots=lots,
        as_of_index=index,
        ledger_id=ledger.ledger_id,
    )


__all__ = [
    "CashBalance",
    "Holding",
    "LedgerTrade",
    "LotLedgerEntry",
    "PortfolioLedger",
    "PortfolioLedgerSnapshot",
    "TaxLot",
    "portfolio_ledger_snapshot",
]
