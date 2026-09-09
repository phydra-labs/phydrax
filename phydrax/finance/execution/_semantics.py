#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable host-side execution records and exact event-ledger transitions."""

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from numbers import Integral
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..core import Currency, InstrumentReference


class Side(IntEnum):
    """Signed inventory direction of an order."""

    SELL = -1
    BUY = 1


class OrderKind(IntEnum):
    """Supported semantic order instructions; no venue behavior is implied."""

    MARKET = 0
    LIMIT = 1


class OrderStatus(IntEnum):
    """Lifecycle status derived exclusively from accepted ledger events."""

    ACTIVE = 0
    PARTIALLY_FILLED = 1
    FILLED = 2
    CANCELLED = 3


class FillStatus(IntEnum):
    """Whether one accepted fill leaves quantity resting or completes its order."""

    PARTIAL = 0
    COMPLETE = 1


class ExecutionEventKind(IntEnum):
    """Total-order event variants for an externally supplied execution history."""

    SUBMIT = 0
    FILL = 1
    CANCEL = 2


ExecutionEventOrder: TypeAlias = Literal["event-time-then-sequence"]


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _nonnegative_int(value: int, owner: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{owner} must be a nonnegative integer.")
    return int(value)


def _positive_scalar(value: ArrayLike, owner: str, /) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != ():
        raise ValueError(f"{owner} must be scalar.")
    if not (
        jnp.issubdtype(scalar.dtype, jnp.number)
        and not jnp.issubdtype(scalar.dtype, jnp.complexfloating)
    ):
        raise TypeError(f"{owner} must be real-valued.")
    scalar = scalar if jnp.issubdtype(scalar.dtype, jnp.inexact) else scalar.astype(float)
    if not bool(jnp.isfinite(scalar)) or not bool(scalar > 0.0):
        raise ValueError(f"{owner} must be finite and positive.")
    return scalar


def _nonnegative_scalar(value: ArrayLike, owner: str, /) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != ():
        raise ValueError(f"{owner} must be scalar.")
    if not (
        jnp.issubdtype(scalar.dtype, jnp.number)
        and not jnp.issubdtype(scalar.dtype, jnp.complexfloating)
    ):
        raise TypeError(f"{owner} must be real-valued.")
    scalar = scalar if jnp.issubdtype(scalar.dtype, jnp.inexact) else scalar.astype(float)
    if not bool(jnp.isfinite(scalar)) or bool(scalar < 0.0):
        raise ValueError(f"{owner} must be finite and nonnegative.")
    return scalar


def _finite_scalar(value: ArrayLike, owner: str, /) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != ():
        raise ValueError(f"{owner} must be scalar.")
    if not (
        jnp.issubdtype(scalar.dtype, jnp.number)
        and not jnp.issubdtype(scalar.dtype, jnp.complexfloating)
    ):
        raise TypeError(f"{owner} must be real-valued.")
    scalar = scalar if jnp.issubdtype(scalar.dtype, jnp.inexact) else scalar.astype(float)
    if not bool(jnp.isfinite(scalar)):
        raise ValueError(f"{owner} must be finite.")
    return scalar


class ExecutionOrder(StrictModule):
    """One immutable client instruction, independent of any matching mechanism."""

    instrument: InstrumentReference
    quantity: Array
    limit_price: Array | None
    order_id: str = eqx.field(static=True)
    side: Side = eqx.field(static=True)
    kind: OrderKind = eqx.field(static=True)
    submitted_ns: int = eqx.field(static=True)
    expires_ns: int | None = eqx.field(static=True)
    sequence: int = eqx.field(static=True)

    def __init__(
        self,
        instrument: InstrumentReference,
        quantity: ArrayLike,
        /,
        *,
        order_id: str,
        side: Side,
        kind: OrderKind,
        submitted_ns: int,
        sequence: int,
        limit_price: ArrayLike | None = None,
        expires_ns: int | None = None,
    ):
        if not isinstance(instrument, InstrumentReference):
            raise TypeError("instrument must be an InstrumentReference.")
        if not isinstance(side, Side):
            raise TypeError("side must be Side.BUY or Side.SELL.")
        if not isinstance(kind, OrderKind):
            raise TypeError("kind must be OrderKind.MARKET or OrderKind.LIMIT.")
        side_value = side
        kind_value = kind
        submitted = _nonnegative_int(submitted_ns, "submitted_ns")
        expires = (
            None if expires_ns is None else _nonnegative_int(expires_ns, "expires_ns")
        )
        if expires is not None and expires < submitted:
            raise ValueError("expires_ns cannot precede submitted_ns.")
        if kind_value == OrderKind.LIMIT and limit_price is None:
            raise ValueError("A limit order requires limit_price.")
        if kind_value == OrderKind.MARKET and limit_price is not None:
            raise ValueError("A market order cannot carry limit_price.")
        self.instrument = instrument
        self.quantity = _positive_scalar(quantity, "quantity")
        self.limit_price = (
            None if limit_price is None else _positive_scalar(limit_price, "limit_price")
        )
        self.order_id = _identifier(order_id, "order_id")
        self.side = side_value
        self.kind = kind_value
        self.submitted_ns = submitted
        self.expires_ns = expires
        self.sequence = _nonnegative_int(sequence, "sequence")


class ExecutionFill(StrictModule):
    """An externally supplied fill observation applied exactly once by a ledger."""

    quantity: Array
    price: Array
    fee: Array
    fill_id: str = eqx.field(static=True)
    order_id: str = eqx.field(static=True)
    executed_ns: int = eqx.field(static=True)
    sequence: int = eqx.field(static=True)

    def __init__(
        self,
        quantity: ArrayLike,
        price: ArrayLike,
        /,
        *,
        fill_id: str,
        order_id: str,
        executed_ns: int,
        sequence: int,
        fee: ArrayLike = 0.0,
    ):
        self.quantity = _positive_scalar(quantity, "quantity")
        self.price = _positive_scalar(price, "price")
        self.fee = _nonnegative_scalar(fee, "fee")
        self.fill_id = _identifier(fill_id, "fill_id")
        self.order_id = _identifier(order_id, "order_id")
        self.executed_ns = _nonnegative_int(executed_ns, "executed_ns")
        self.sequence = _nonnegative_int(sequence, "sequence")


class ExecutionInventory(StrictModule):
    """Cash, signed inventory, and fee totals after an accepted event prefix."""

    instrument: InstrumentReference
    currency: Currency
    cash: Array
    quantity: Array
    fees: Array

    def __init__(
        self,
        instrument: InstrumentReference,
        currency: Currency,
        /,
        *,
        cash: ArrayLike = 0.0,
        quantity: ArrayLike = 0.0,
        fees: ArrayLike = 0.0,
    ):
        if not isinstance(instrument, InstrumentReference):
            raise TypeError("instrument must be an InstrumentReference.")
        if not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency.")
        if instrument.settlement_currency.currency_id != currency.currency_id:
            raise ValueError(
                "inventory currency must equal the instrument settlement currency."
            )
        self.instrument = instrument
        self.currency = currency
        self.cash = _finite_scalar(cash, "cash")
        self.quantity = _finite_scalar(quantity, "quantity")
        self.fees = _nonnegative_scalar(fees, "fees")

    def marked_value(self, mark_price: ArrayLike, /) -> Array:
        """Return cash plus signed inventory valued at one positive supplied mark."""

        mark = _positive_scalar(mark_price, "mark_price")
        return self.cash + self.quantity * mark


class ExecutionAction(StrictModule):
    """A predictable execution action; acceptance never implies order matching."""

    market_quantity: Array
    trading_rate: Array
    bid_offset: Array
    ask_offset: Array
    impulse_quantity: Array
    cancel_active: Array
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        market_quantity: ArrayLike = 0.0,
        trading_rate: ArrayLike = 0.0,
        bid_offset: ArrayLike = 0.0,
        ask_offset: ArrayLike = 0.0,
        impulse_quantity: ArrayLike = 0.0,
        cancel_active: bool = False,
        action_id: str,
    ):
        self.market_quantity = _finite_scalar(market_quantity, "market_quantity")
        self.trading_rate = _finite_scalar(trading_rate, "trading_rate")
        self.bid_offset = _nonnegative_scalar(bid_offset, "bid_offset")
        self.ask_offset = _nonnegative_scalar(ask_offset, "ask_offset")
        self.impulse_quantity = _finite_scalar(impulse_quantity, "impulse_quantity")
        cancel = jnp.asarray(cancel_active, dtype=bool)
        if cancel.shape != ():
            raise ValueError("cancel_active must be scalar.")
        self.cancel_active = cancel
        self.action_id = _identifier(action_id, "action_id")

    @property
    def vector(self) -> Array:
        return jnp.stack(
            (
                self.market_quantity,
                self.trading_rate,
                self.bid_offset,
                self.ask_offset,
                self.impulse_quantity,
                self.cancel_active.astype(self.market_quantity.dtype),
            )
        )


class ExecutionState(StrictModule):
    """Fixed-shape numerical execution state used by model/control adapters."""

    cash: Array
    inventory: Array
    remaining_parent: Array
    mid_price: Array
    permanent_impact: Array
    transient_impact: Array
    queue_depth: Array
    hawkes_excitation: Array
    num_queues: int = eqx.field(static=True)
    num_hawkes_channels: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        cash: ArrayLike,
        inventory: ArrayLike,
        remaining_parent: ArrayLike,
        mid_price: ArrayLike,
        permanent_impact: ArrayLike = 0.0,
        transient_impact: ArrayLike = 0.0,
        queue_depth: ArrayLike,
        hawkes_excitation: ArrayLike,
    ):
        queue = jnp.asarray(queue_depth)
        excitation = jnp.asarray(hawkes_excitation)
        if queue.ndim != 1:
            raise ValueError("queue_depth must be a rank-one vector.")
        if excitation.ndim != 1:
            raise ValueError("hawkes_excitation must be a rank-one vector.")
        if jnp.issubdtype(queue.dtype, jnp.complexfloating) or jnp.issubdtype(
            excitation.dtype, jnp.complexfloating
        ):
            raise TypeError("queue_depth and hawkes_excitation must be real-valued.")
        if not bool(jnp.all(jnp.isfinite(queue))) or bool(jnp.any(queue < 0.0)):
            raise ValueError("queue_depth must be finite and nonnegative.")
        if not bool(jnp.all(jnp.isfinite(excitation))) or bool(jnp.any(excitation < 0.0)):
            raise ValueError("hawkes_excitation must be finite and nonnegative.")
        self.cash = _finite_scalar(cash, "cash")
        self.inventory = _finite_scalar(inventory, "inventory")
        self.remaining_parent = _nonnegative_scalar(remaining_parent, "remaining_parent")
        self.mid_price = _positive_scalar(mid_price, "mid_price")
        self.permanent_impact = _finite_scalar(permanent_impact, "permanent_impact")
        self.transient_impact = _finite_scalar(transient_impact, "transient_impact")
        self.queue_depth = queue.astype(float)
        self.hawkes_excitation = excitation.astype(float)
        self.num_queues = int(queue.size)
        self.num_hawkes_channels = int(excitation.size)

    @property
    def vector(self) -> Array:
        return jnp.concatenate(
            (
                jnp.stack(
                    (
                        self.cash,
                        self.inventory,
                        self.remaining_parent,
                        self.mid_price,
                        self.permanent_impact,
                        self.transient_impact,
                    )
                ),
                self.queue_depth,
                self.hawkes_excitation,
            )
        )


class ExecutionConstraints(StrictModule):
    """Explicit hard action/state bounds; invalid actions are never clipped."""

    maximum_absolute_inventory: float = eqx.field(static=True)
    maximum_order_quantity: float = eqx.field(static=True)
    maximum_absolute_rate: float = eqx.field(static=True)
    minimum_quote_offset: float = eqx.field(static=True)
    maximum_quote_offset: float = eqx.field(static=True)
    require_terminal_flat: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_absolute_inventory: float,
        maximum_order_quantity: float,
        maximum_absolute_rate: float,
        minimum_quote_offset: float = 0.0,
        maximum_quote_offset: float,
        require_terminal_flat: bool = False,
    ):
        values = (
            maximum_absolute_inventory,
            maximum_order_quantity,
            maximum_absolute_rate,
            minimum_quote_offset,
            maximum_quote_offset,
        )
        if any(not isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise ValueError(
                "Execution constraint bounds must be finite and nonnegative."
            )
        if maximum_quote_offset < minimum_quote_offset:
            raise ValueError("maximum_quote_offset cannot be below minimum_quote_offset.")
        self.maximum_absolute_inventory = float(maximum_absolute_inventory)
        self.maximum_order_quantity = float(maximum_order_quantity)
        self.maximum_absolute_rate = float(maximum_absolute_rate)
        self.minimum_quote_offset = float(minimum_quote_offset)
        self.maximum_quote_offset = float(maximum_quote_offset)
        self.require_terminal_flat = bool(require_terminal_flat)

    def action_is_admissible(
        self, action: ExecutionAction, state: ExecutionState, /
    ) -> Array:
        if not isinstance(action, ExecutionAction):
            raise TypeError("action must be an ExecutionAction.")
        if not isinstance(state, ExecutionState):
            raise TypeError("state must be an ExecutionState.")
        proposed_inventory = (
            state.inventory + action.market_quantity + action.impulse_quantity
        )
        return (
            (jnp.abs(action.market_quantity) <= self.maximum_order_quantity)
            & (jnp.abs(action.impulse_quantity) <= self.maximum_order_quantity)
            & (jnp.abs(action.trading_rate) <= self.maximum_absolute_rate)
            & (action.bid_offset >= self.minimum_quote_offset)
            & (action.bid_offset <= self.maximum_quote_offset)
            & (action.ask_offset >= self.minimum_quote_offset)
            & (action.ask_offset <= self.maximum_quote_offset)
            & (jnp.abs(proposed_inventory) <= self.maximum_absolute_inventory)
        )


class ExecutionEvent(StrictModule):
    """One totally ordered submit, fill, or cancel event."""

    order: ExecutionOrder | None
    fill: ExecutionFill | None
    event_id: str = eqx.field(static=True)
    kind: ExecutionEventKind = eqx.field(static=True)
    event_time_ns: int = eqx.field(static=True)
    sequence: int = eqx.field(static=True)
    order_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        event_id: str,
        kind: ExecutionEventKind,
        event_time_ns: int,
        sequence: int,
        order: ExecutionOrder | None = None,
        fill: ExecutionFill | None = None,
        order_id: str | None = None,
    ):
        if not isinstance(kind, ExecutionEventKind):
            raise TypeError("kind must be a supported ExecutionEventKind.")
        kind_value = kind
        event_time = _nonnegative_int(event_time_ns, "event_time_ns")
        sequence_value = _nonnegative_int(sequence, "sequence")
        if kind_value == ExecutionEventKind.SUBMIT:
            if order is None or fill is not None or order_id is not None:
                raise ValueError("SUBMIT requires only order.")
            if order.submitted_ns != event_time or order.sequence != sequence_value:
                raise ValueError("SUBMIT event ordering must equal its order ordering.")
        elif kind_value == ExecutionEventKind.FILL:
            if fill is None or order is not None or order_id is not None:
                raise ValueError("FILL requires only fill.")
            if fill.executed_ns != event_time or fill.sequence != sequence_value:
                raise ValueError("FILL event ordering must equal its fill ordering.")
        else:
            if order_id is None or order is not None or fill is not None:
                raise ValueError("CANCEL requires only order_id.")
            _identifier(order_id, "order_id")
        self.order = order
        self.fill = fill
        self.event_id = _identifier(event_id, "event_id")
        self.kind = kind_value
        self.event_time_ns = event_time
        self.sequence = sequence_value
        self.order_id = order_id

    @property
    def ordering_key(self) -> tuple[int, int]:
        return self.event_time_ns, self.sequence


class ExecutionLedger(StrictModule):
    """Append-only accepted events and their exact derived order/inventory state."""

    initial_inventory: ExecutionInventory
    inventory: ExecutionInventory
    orders: tuple[ExecutionOrder, ...]
    order_statuses: tuple[OrderStatus, ...] = eqx.field(static=True)
    remaining_quantities: Array
    fills: tuple[ExecutionFill, ...]
    fill_statuses: tuple[FillStatus, ...] = eqx.field(static=True)
    events: tuple[ExecutionEvent, ...]
    event_order: ExecutionEventOrder = eqx.field(static=True)
    quantity_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        initial_inventory: ExecutionInventory,
        /,
        *,
        quantity_tolerance: float = 1.0e-10,
    ):
        if not isinstance(initial_inventory, ExecutionInventory):
            raise TypeError("initial_inventory must be an ExecutionInventory.")
        tolerance = float(quantity_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("quantity_tolerance must be finite and nonnegative.")
        self.initial_inventory = initial_inventory
        self.inventory = initial_inventory
        self.orders = ()
        self.order_statuses = ()
        self.remaining_quantities = jnp.zeros(
            (0,), dtype=initial_inventory.quantity.dtype
        )
        self.fills = ()
        self.fill_statuses = ()
        self.events = ()
        self.event_order = "event-time-then-sequence"
        self.quantity_tolerance = tolerance


class ExecutionAccounting(StrictModule):
    """Independent marked wealth and PnL decomposition for one ledger."""

    cash: Array
    inventory: Array
    inventory_value: Array
    marked_wealth: Array
    initial_marked_wealth: Array
    gross_trading_pnl: Array
    fees: Array
    net_pnl: Array
    currency: Currency


class ExecutionReplayEvidence(StrictModule):
    """Independent ledger reconstruction residuals for cash, inventory, and PnL."""

    replayed_ledger: ExecutionLedger
    cash_residual: Array
    inventory_residual: Array
    pnl_residual: Array
    event_count_matches: Array
    passed: Array
    scope: str = eqx.field(static=True)


def _order_index(ledger: ExecutionLedger, order_id: str, /) -> int:
    for index, order in enumerate(ledger.orders):
        if order.order_id == order_id:
            return index
    raise ValueError(f"Unknown order_id {order_id!r}.")


def _replace_tuple(values: tuple, index: int, value, /) -> tuple:
    return values[:index] + (value,) + values[index + 1 :]


def _ledger_from_parts(
    ledger: ExecutionLedger,
    *,
    inventory: ExecutionInventory,
    orders: tuple[ExecutionOrder, ...],
    order_statuses: tuple[OrderStatus, ...],
    remaining_quantities: Array,
    fills: tuple[ExecutionFill, ...],
    fill_statuses: tuple[FillStatus, ...],
    events: tuple[ExecutionEvent, ...],
) -> ExecutionLedger:
    rebuilt = ExecutionLedger(
        ledger.initial_inventory,
        quantity_tolerance=ledger.quantity_tolerance,
    )
    object.__setattr__(rebuilt, "inventory", inventory)
    object.__setattr__(rebuilt, "orders", orders)
    object.__setattr__(rebuilt, "order_statuses", order_statuses)
    object.__setattr__(rebuilt, "remaining_quantities", remaining_quantities)
    object.__setattr__(rebuilt, "fills", fills)
    object.__setattr__(rebuilt, "fill_statuses", fill_statuses)
    object.__setattr__(rebuilt, "events", events)
    return rebuilt


def apply_execution_event(
    ledger: ExecutionLedger, event: ExecutionEvent, /
) -> ExecutionLedger:
    """Validate and append one event, rejecting every duplicate or illegal transition."""

    if not isinstance(ledger, ExecutionLedger):
        raise TypeError("ledger must be an ExecutionLedger.")
    if not isinstance(event, ExecutionEvent):
        raise TypeError("event must be an ExecutionEvent.")
    if any(existing.event_id == event.event_id for existing in ledger.events):
        raise ValueError(f"Duplicate event_id {event.event_id!r}.")
    if ledger.events and event.ordering_key <= ledger.events[-1].ordering_key:
        raise ValueError(
            "Execution events must be strictly ordered by event_time_ns then sequence."
        )

    orders = ledger.orders
    statuses = ledger.order_statuses
    remaining = ledger.remaining_quantities
    fills = ledger.fills
    fill_statuses = ledger.fill_statuses
    inventory = ledger.inventory
    if event.kind == ExecutionEventKind.SUBMIT:
        order = event.order
        if order is None:
            raise ValueError("SUBMIT event has no order payload.")
        if any(existing.order_id == order.order_id for existing in orders):
            raise ValueError(f"Duplicate order_id {order.order_id!r}.")
        if order.instrument.instrument_id != inventory.instrument.instrument_id:
            raise ValueError("Order instrument does not match the ledger inventory.")
        orders = orders + (order,)
        statuses = statuses + (OrderStatus.ACTIVE,)
        remaining = jnp.concatenate((remaining, order.quantity[None]))
    elif event.kind == ExecutionEventKind.FILL:
        fill = event.fill
        if fill is None:
            raise ValueError("FILL event has no fill payload.")
        if any(existing.fill_id == fill.fill_id for existing in fills):
            raise ValueError(f"Duplicate fill_id {fill.fill_id!r}.")
        index = _order_index(ledger, fill.order_id)
        status = statuses[index]
        if status not in (OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED):
            raise ValueError("A fill requires an active or partially filled order.")
        order = orders[index]
        if fill.executed_ns < order.submitted_ns:
            raise ValueError("A fill cannot precede order submission.")
        if order.expires_ns is not None and fill.executed_ns > order.expires_ns:
            raise ValueError("A fill cannot follow order expiry.")
        if order.kind == OrderKind.LIMIT:
            limit_price = order.limit_price
            if limit_price is None:
                raise ValueError("Limit order has no limit price.")
            if order.side == Side.BUY and bool(
                fill.price > limit_price + ledger.quantity_tolerance
            ):
                raise ValueError("Buy fill price exceeds the order limit price.")
            if order.side == Side.SELL and bool(
                fill.price < limit_price - ledger.quantity_tolerance
            ):
                raise ValueError("Sell fill price is below the order limit price.")
        available = remaining[index]
        if bool(fill.quantity > available + ledger.quantity_tolerance):
            raise ValueError("Fill quantity exceeds the remaining order quantity.")
        residual = jnp.maximum(available - fill.quantity, 0.0)
        remaining = remaining.at[index].set(residual)
        next_status = (
            OrderStatus.FILLED
            if bool(residual <= ledger.quantity_tolerance)
            else OrderStatus.PARTIALLY_FILLED
        )
        statuses = _replace_tuple(statuses, index, next_status)
        signed_quantity = int(order.side) * fill.quantity
        inventory = ExecutionInventory(
            inventory.instrument,
            inventory.currency,
            cash=inventory.cash - signed_quantity * fill.price - fill.fee,
            quantity=inventory.quantity + signed_quantity,
            fees=inventory.fees + fill.fee,
        )
        fills = fills + (fill,)
        fill_statuses = fill_statuses + (
            FillStatus.COMPLETE
            if next_status == OrderStatus.FILLED
            else FillStatus.PARTIAL,
        )
    else:
        order_id = event.order_id
        if order_id is None:
            raise ValueError("CANCEL event has no order identity.")
        index = _order_index(ledger, order_id)
        if statuses[index] not in (OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED):
            raise ValueError("Only an active or partially filled order may be cancelled.")
        statuses = _replace_tuple(statuses, index, OrderStatus.CANCELLED)

    return _ledger_from_parts(
        ledger,
        inventory=inventory,
        orders=orders,
        order_statuses=statuses,
        remaining_quantities=remaining,
        fills=fills,
        fill_statuses=fill_statuses,
        events=ledger.events + (event,),
    )


def execution_cash_inventory_pnl(
    ledger: ExecutionLedger,
    /,
    *,
    initial_mark_price: ArrayLike,
    terminal_mark_price: ArrayLike,
) -> ExecutionAccounting:
    """Compute marked wealth and net PnL without reusing a model objective."""

    if not isinstance(ledger, ExecutionLedger):
        raise TypeError("ledger must be an ExecutionLedger.")
    initial_mark = _positive_scalar(initial_mark_price, "initial_mark_price")
    terminal_mark = _positive_scalar(terminal_mark_price, "terminal_mark_price")
    inventory_value = ledger.inventory.quantity * terminal_mark
    wealth = ledger.inventory.cash + inventory_value
    initial_wealth = (
        ledger.initial_inventory.cash + ledger.initial_inventory.quantity * initial_mark
    )
    net = wealth - initial_wealth
    incremental_fees = ledger.inventory.fees - ledger.initial_inventory.fees
    return ExecutionAccounting(
        cash=ledger.inventory.cash,
        inventory=ledger.inventory.quantity,
        inventory_value=inventory_value,
        marked_wealth=wealth,
        initial_marked_wealth=initial_wealth,
        gross_trading_pnl=net + incremental_fees,
        fees=incremental_fees,
        net_pnl=net,
        currency=ledger.inventory.currency,
    )


def replay_execution_ledger(
    ledger: ExecutionLedger,
    /,
    *,
    initial_mark_price: ArrayLike,
    terminal_mark_price: ArrayLike,
    tolerance: float = 1.0e-10,
) -> ExecutionReplayEvidence:
    """Independently rebuild events and compare cash, inventory, and marked PnL."""

    if not isinstance(ledger, ExecutionLedger):
        raise TypeError("ledger must be an ExecutionLedger.")
    threshold = float(tolerance)
    if not isfinite(threshold) or threshold < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    replayed = ExecutionLedger(
        ledger.initial_inventory,
        quantity_tolerance=ledger.quantity_tolerance,
    )
    for event in ledger.events:
        replayed = apply_execution_event(replayed, event)
    reported = execution_cash_inventory_pnl(
        ledger,
        initial_mark_price=initial_mark_price,
        terminal_mark_price=terminal_mark_price,
    )
    terminal_mark = _positive_scalar(terminal_mark_price, "terminal_mark_price")
    independent_cash = ledger.initial_inventory.cash
    independent_inventory = ledger.initial_inventory.quantity
    side_by_order_id: dict[str, Side] = {}
    for event in ledger.events:
        if event.kind == ExecutionEventKind.SUBMIT:
            order = event.order
            if order is None:
                raise ValueError("SUBMIT event has no order payload.")
            side_by_order_id[order.order_id] = order.side
        elif event.kind == ExecutionEventKind.FILL:
            fill = event.fill
            if fill is None:
                raise ValueError("FILL event has no fill payload.")
            if fill.order_id not in side_by_order_id:
                raise ValueError("Replay fill precedes its order submission.")
            signed_quantity = int(side_by_order_id[fill.order_id]) * fill.quantity
            independent_inventory = independent_inventory + signed_quantity
            independent_cash = independent_cash - signed_quantity * fill.price - fill.fee
    independent_pnl = (
        independent_cash
        + independent_inventory * terminal_mark
        - reported.initial_marked_wealth
    )
    cash_residual = jnp.abs(ledger.inventory.cash - independent_cash)
    inventory_residual = jnp.abs(ledger.inventory.quantity - independent_inventory)
    pnl_residual = jnp.abs(reported.net_pnl - independent_pnl)
    count_matches = jnp.asarray(len(ledger.events) == len(replayed.events))
    passed = (
        (cash_residual <= threshold)
        & (inventory_residual <= threshold)
        & (pnl_residual <= threshold)
        & count_matches
    )
    return ExecutionReplayEvidence(
        replayed_ledger=replayed,
        cash_residual=cash_residual,
        inventory_residual=inventory_residual,
        pnl_residual=pnl_residual,
        event_count_matches=count_matches,
        passed=passed,
        scope="independent-event-prefix-cash-inventory-marked-pnl-reconstruction",
    )


def require_admissible_execution_action(
    action: ExecutionAction,
    state: ExecutionState,
    constraints: ExecutionConstraints,
    /,
) -> None:
    """Reject, rather than silently project, an inadmissible execution action."""

    if not isinstance(constraints, ExecutionConstraints):
        raise TypeError("constraints must be ExecutionConstraints.")
    if not bool(constraints.action_is_admissible(action, state)):
        raise ValueError("Execution action violates the declared hard constraints.")


__all__ = [
    "ExecutionAccounting",
    "ExecutionAction",
    "ExecutionConstraints",
    "ExecutionEvent",
    "ExecutionEventKind",
    "ExecutionEventOrder",
    "ExecutionFill",
    "FillStatus",
    "ExecutionInventory",
    "ExecutionLedger",
    "ExecutionOrder",
    "ExecutionReplayEvidence",
    "ExecutionState",
    "OrderKind",
    "OrderStatus",
    "Side",
    "apply_execution_event",
    "execution_cash_inventory_pnl",
    "replay_execution_ledger",
    "require_admissible_execution_action",
]
