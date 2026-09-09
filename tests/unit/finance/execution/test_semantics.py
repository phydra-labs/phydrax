#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.finance.core import Currency, FinancialIdentifier, InstrumentReference
from phydrax.finance.execution._semantics import (
    apply_execution_event,
    execution_cash_inventory_pnl,
    ExecutionEvent,
    ExecutionEventKind,
    ExecutionFill,
    ExecutionInventory,
    ExecutionLedger,
    ExecutionOrder,
    FillStatus,
    OrderKind,
    OrderStatus,
    replay_execution_ledger,
    Side,
)


def _instrument():
    usd = Currency("USD", 2)
    instrument = InstrumentReference(
        FinancialIdentifier("test", "ABC"),
        (),
        usd,
        "USD_PER_SHARE",
        "Test instrument",
    )
    return usd, instrument


def _submitted_order(instrument):
    return ExecutionOrder(
        instrument,
        10.0,
        order_id="order-1",
        side=Side.BUY,
        kind=OrderKind.LIMIT,
        limit_price=10.0,
        submitted_ns=1,
        sequence=0,
    )


def test_submit_partial_fill_cancel_conserves_cash_inventory_and_replays():
    usd, instrument = _instrument()
    ledger = ExecutionLedger(ExecutionInventory(instrument, usd))
    order = _submitted_order(instrument)
    ledger = apply_execution_event(
        ledger,
        ExecutionEvent(
            event_id="submit",
            kind=ExecutionEventKind.SUBMIT,
            event_time_ns=1,
            sequence=0,
            order=order,
        ),
    )
    fill = ExecutionFill(
        4.0,
        10.0,
        fee=1.0,
        fill_id="fill-1",
        order_id=order.order_id,
        executed_ns=1,
        sequence=1,
    )
    ledger = apply_execution_event(
        ledger,
        ExecutionEvent(
            event_id="partial-fill",
            kind=ExecutionEventKind.FILL,
            event_time_ns=1,
            sequence=1,
            fill=fill,
        ),
    )
    ledger = apply_execution_event(
        ledger,
        ExecutionEvent(
            event_id="cancel",
            kind=ExecutionEventKind.CANCEL,
            event_time_ns=1,
            sequence=2,
            order_id=order.order_id,
        ),
    )

    assert ledger.order_statuses == (OrderStatus.CANCELLED,)
    assert ledger.fill_statuses == (FillStatus.PARTIAL,)
    np.testing.assert_allclose(ledger.remaining_quantities, jnp.asarray([6.0]))
    assert ledger.inventory.quantity == 4.0
    assert ledger.inventory.cash == -41.0
    accounting = execution_cash_inventory_pnl(
        ledger, initial_mark_price=10.0, terminal_mark_price=10.0
    )
    assert accounting.gross_trading_pnl == 0.0
    assert accounting.fees == 1.0
    assert accounting.net_pnl == -1.0
    replay = replay_execution_ledger(
        ledger, initial_mark_price=10.0, terminal_mark_price=10.0
    )
    assert bool(replay.passed)
    assert replay.cash_residual == 0.0
    assert replay.inventory_residual == 0.0
    assert replay.pnl_residual == 0.0


def test_duplicate_overfill_cancel_and_ambiguous_ordering_fail():
    usd, instrument = _instrument()
    ledger = ExecutionLedger(ExecutionInventory(instrument, usd))
    order = _submitted_order(instrument)
    submit = ExecutionEvent(
        event_id="submit",
        kind=ExecutionEventKind.SUBMIT,
        event_time_ns=1,
        sequence=0,
        order=order,
    )
    ledger = apply_execution_event(ledger, submit)
    with pytest.raises(ValueError, match="Duplicate event_id"):
        apply_execution_event(ledger, submit)

    overfill = ExecutionFill(
        11.0,
        10.0,
        fill_id="too-large",
        order_id=order.order_id,
        executed_ns=1,
        sequence=1,
    )
    with pytest.raises(ValueError, match="exceeds"):
        apply_execution_event(
            ledger,
            ExecutionEvent(
                event_id="overfill",
                kind=ExecutionEventKind.FILL,
                event_time_ns=1,
                sequence=1,
                fill=overfill,
            ),
        )

    ledger = apply_execution_event(
        ledger,
        ExecutionEvent(
            event_id="cancel",
            kind=ExecutionEventKind.CANCEL,
            event_time_ns=1,
            sequence=1,
            order_id=order.order_id,
        ),
    )
    with pytest.raises(ValueError, match="active or partially filled"):
        apply_execution_event(
            ledger,
            ExecutionEvent(
                event_id="cancel-again",
                kind=ExecutionEventKind.CANCEL,
                event_time_ns=1,
                sequence=2,
                order_id=order.order_id,
            ),
        )

    with pytest.raises(ValueError, match="strictly ordered"):
        apply_execution_event(
            ledger,
            ExecutionEvent(
                event_id="same-ordering-key",
                kind=ExecutionEventKind.CANCEL,
                event_time_ns=1,
                sequence=1,
                order_id=order.order_id,
            ),
        )
