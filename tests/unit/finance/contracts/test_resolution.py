#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

from phydrax.finance.contracts._base import AbstractContract, AbstractResolvedContract
from phydrax.finance.contracts._cashflows import CashflowBatch, CashflowStatus
from phydrax.finance.contracts._exercise import SettlementTerms, SettlementType
from phydrax.finance.contracts._resolution import (
    ContractResolutionContext,
    ContractResolutionPlan,
    ContractResolutionStatus,
    ResolvedContract,
)
from phydrax.finance.contracts._trade import Position, Trade
from phydrax.finance.core import (
    BusinessDayRule,
    CalendarSnapshot,
    Currency,
    FinanceDate,
    FinancialTimestamp,
    TemporalAdmissibilityPolicy,
)
from phydrax.finance.market._lineage import DataLineage
from phydrax.finance.market._snapshots import ReferenceDataSnapshot


USD = Currency("USD", 2)
_POLICY = TemporalAdmissibilityPolicy(True, True, True)


def _time(value: int, vintage: str) -> FinancialTimestamp:
    return FinancialTimestamp(value, value, value, value, vintage, _POLICY)


def _settlement() -> SettlementTerms:
    return SettlementTerms(
        USD,
        calendar_id="nyc",
        business_day_rule=BusinessDayRule.FOLLOWING,
        settlement_type=SettlementType.CASH,
    )


class _Contract(AbstractContract):
    contract_id: str = eqx.field(static=True)
    cashflows: CashflowBatch
    settlement: SettlementTerms

    def __init__(
        self, contract_id: str, cashflows: CashflowBatch, settlement: SettlementTerms
    ) -> None:
        self.contract_id = contract_id
        self.cashflows = cashflows
        self.settlement = settlement

    def resolve(self, context: ContractResolutionContext, /) -> AbstractResolvedContract:
        status = context.context_status
        return ResolvedContract(
            self,
            self.cashflows,
            self.settlement,
            resolution_status=status,
        )


def _cashflows() -> CashflowBatch:
    return CashflowBatch(
        (FinanceDate.from_iso("2027-01-15"), FinanceDate.from_iso("2027-07-15")),
        jnp.asarray([5.0, 105.0]),
        (USD, USD),
        obligation_ids=("coupon", "redemption"),
    )


def _context() -> ContractResolutionContext:
    as_of = _time(100, "as-of")
    reference = ReferenceDataSnapshot(
        (),
        (),
        (USD,),
        as_of=as_of,
        lineage=DataLineage("reference-master", "security-master"),
    )
    calendar = CalendarSnapshot("nyc", (), (5, 6), "unit-test")
    return ContractResolutionContext(
        reference,
        (calendar,),
        as_of,
        cashflow_capacity=4,
    )


def test_resolution_scenario_1() -> None:
    cashflows = _cashflows().pad(4)

    assert cashflows.active_count == 2
    assert jnp.array_equal(cashflows.valid_mask, jnp.asarray([True, True, False, False]))
    assert jnp.array_equal(cashflows.amounts, jnp.asarray([5.0, 105.0, 0.0, 0.0]))
    assert jnp.array_equal(
        cashflows.payment_ordinals[2:], jnp.zeros((2,), dtype=jnp.int32)
    )
    assert jnp.array_equal(cashflows.currency_index[2:], jnp.zeros((2,), dtype=jnp.int32))
    assert cashflows.obligation_ids[2:] == ("", "")
    prepared = _cashflows().prepare(1)

    assert int(prepared.preparation_status) == int(CashflowStatus.CAPACITY_EXCEEDED)
    assert not bool(jnp.any(prepared.accepted))
    assert prepared.cashflows.capacity == 1
    terms = SettlementTerms(
        USD,
        settlement_lag_days=2,
        calendar_id="nyc",
        business_day_rule=BusinessDayRule.FOLLOWING,
    )
    calendar = CalendarSnapshot("nyc", (), (5, 6), "unit-test")

    settlement_date = terms.resolve_date(
        FinanceDate.from_iso("2027-01-08"),
        calendar,
    )

    assert settlement_date.isoformat() == "2027-01-12"
    with pytest.raises(ValueError, match="identifiers"):
        terms.resolve_date(
            FinanceDate.from_iso("2027-01-08"),
            CalendarSnapshot("lon", (), (5, 6), "unit-test"),
        )
    definition = _Contract("bond-1", _cashflows(), _settlement())
    plan = ContractResolutionPlan(definition, _context())

    first = plan.resolve()
    second = plan.resolve()
    replay = plan.replay(first.resolved_id)

    assert first.resolved_id == second.resolved_id
    assert first.contract_id == definition.contract_id
    assert bool(replay.matches)
    assert int(replay.status) == int(ContractResolutionStatus.SUCCESS)
    assert replay.resolved.resolved_id == first.resolved_id

    mismatch = plan.replay("different-resolution")
    assert not bool(mismatch.matches)
    assert int(mismatch.status) == int(ContractResolutionStatus.REPLAY_MISMATCH)


def test_resolution_scenario_2() -> None:
    reference = ReferenceDataSnapshot(
        (),
        (),
        (USD,),
        as_of=_time(200, "future-reference"),
        lineage=DataLineage("reference-master", "future-security-master"),
    )
    context = ContractResolutionContext(
        reference,
        (CalendarSnapshot("nyc", (), (5, 6), "unit-test"),),
        _time(100, "decision"),
        cashflow_capacity=2,
    )

    assert not context.accepted
    assert context.context_status & ContractResolutionStatus.CAUSAL_TIME_VIOLATION
    definition = _Contract("bond-1", _cashflows(), _settlement())
    resolved = ContractResolutionPlan(definition, _context()).resolve()
    trade = Trade(
        "trade-1",
        definition,
        3.0,
        _time(100, "execution"),
        settlement=_settlement(),
    )
    position = Position("position-1", resolved, 3.0, account_id="book-a")

    assert trade.contract is definition
    assert position.contract is resolved
    assert trade.trade_id != position.position_id
    assert not isinstance(trade, Position)
    assert not isinstance(position, Trade)
    with pytest.raises(TypeError, match="unresolved"):
        Trade(
            "bad-trade",
            resolved,  # ty: ignore[invalid-argument-type]
            1.0,
            _time(100, "bad-execution"),
            settlement=_settlement(),
        )
    with pytest.raises(TypeError, match="resolved"):
        Position(
            "bad-position",
            definition,  # ty: ignore[invalid-argument-type]
            1.0,
            account_id="book-a",
        )
