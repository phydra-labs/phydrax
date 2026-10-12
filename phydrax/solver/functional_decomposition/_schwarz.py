#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from ..._term import AbstractScalarTerm
from ...domain import DomainFunction, PairedSupport
from ...nonlinear import AitkenRelaxation
from ...typing import checked, PRNGKey
from ._problem import FunctionalDecompositionProblem


class TraceExchangeState(StrictModule):
    """Fixed paired trace values and relaxed incoming targets for one interface."""

    points: Any
    left_values: Array
    right_values: Array
    left_target: Array
    right_target: Array
    defect: Array
    pairing_id: str = eqx.field(static=True)
    quantity_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        pairing_id: str,
        points: Any,
        left_values: Array,
        right_values: Array,
        left_target: Array,
        right_target: Array,
        quantity_id: str = "value",
    ) -> None:
        left = jnp.asarray(left_values)
        right = jnp.asarray(right_values)
        left_target_ = jnp.asarray(left_target)
        right_target_ = jnp.asarray(right_target)
        if left.shape != right.shape:
            raise ValueError("Paired trace values must have matching shapes.")
        if left_target_.shape != left.shape or right_target_.shape != right.shape:
            raise ValueError("Relaxed trace targets must match trace value shapes.")
        self.pairing_id = str(pairing_id)
        self.quantity_id = str(quantity_id)
        self.points = points
        self.left_values = left
        self.right_values = right
        self.left_target = left_target_
        self.right_target = right_target_
        self.defect = jnp.max(jnp.abs(right - left))


class SchwarzTraceState(StrictModule):
    """All fixed interface exchanges at one accepted Schwarz sweep boundary."""

    exchanges: tuple[TraceExchangeState, ...]
    maximum_defect: Array
    sweep: int = eqx.field(static=True)

    def __init__(
        self,
        exchanges: Sequence[TraceExchangeState],
        /,
        *,
        sweep: int,
    ) -> None:
        exchanges_ = tuple(exchanges)
        if any(not isinstance(value, TraceExchangeState) for value in exchanges_):
            raise TypeError("exchanges must contain TraceExchangeState objects.")
        self.exchanges = exchanges_
        self.maximum_defect = (
            jnp.max(jnp.stack(tuple(value.defect for value in exchanges_)))
            if exchanges_
            else jnp.asarray(0.0)
        )
        self.sweep = int(sweep)

    def exchange(self, pairing_id: str, /) -> TraceExchangeState:
        for value in self.exchanges:
            if value.pairing_id == pairing_id:
                return value
        raise KeyError(f"Unknown Schwarz pairing {pairing_id!r}.")


class DiscreteTracePenalty(AbstractScalarTerm):
    """One-sided local trace fit against a frozen Schwarz target array."""

    pairing: PairedSupport
    points: Any
    target: Array
    scale: Array
    fields: tuple[str, ...] = eqx.field(static=True)
    side: Literal["left", "right"] = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    @checked
    def __init__(
        self,
        field_name: str,
        pairing: PairedSupport,
        points: Any,
        target: Array,
        /,
        *,
        side: Literal["left", "right"],
        scale: float = 1.0,
        label: str | None = None,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'.")
        target_ = jnp.asarray(target)
        scale_ = jnp.asarray(scale, dtype=jnp.float64).reshape(())
        if float(scale_) < 0.0:
            raise ValueError("scale must be non-negative.")
        self.fields = (str(field_name),)
        self.pairing = pairing
        self.points = points
        self.target = target_
        self.scale = scale_
        self.side = side
        self.label = None if label is None else str(label)

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: PRNGKey = DOC_KEY0,
        iter_: int | Array | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_, kwargs
        trace = self.pairing.trace(
            functions[self.fields[0]],
            side=self.side,
        )(self.points, key=key).data
        residual = jnp.asarray(trace) - self.target
        return self.scale * jnp.mean(jnp.real(jnp.conj(residual) * residual))


def capture_schwarz_trace_state(
    problem: FunctionalDecompositionProblem,
    functions: Mapping[str, DomainFunction],
    batches: Sequence[Any],
    /,
    *,
    sweep: int,
    relaxation: float = 1.0,
    previous: SchwarzTraceState | None = None,
) -> SchwarzTraceState:
    """Capture outgoing traces and relaxed incoming targets on fixed supports."""
    if len(batches) != len(problem.cover.pairings):
        raise ValueError("One fixed trace batch is required per paired support.")
    exchanges = []
    for pairing, points in zip(problem.cover.pairings, batches, strict=True):
        left_name = problem.family.field_name(pairing.left_patch_id)
        right_name = problem.family.field_name(pairing.right_patch_id)
        left = pairing.trace(functions[left_name], side="left")(points).data
        right = pairing.trace(functions[right_name], side="right")(points).data
        if previous is None:
            left_target = right
            right_target = left
        else:
            old = previous.exchange(pairing.pairing_id)
            left_target = (1.0 - relaxation) * old.left_target + relaxation * right
            right_target = (1.0 - relaxation) * old.right_target + relaxation * left
        exchanges.append(
            TraceExchangeState(
                pairing_id=pairing.pairing_id,
                points=points,
                left_values=left,
                right_values=right,
                left_target=left_target,
                right_target=right_target,
            )
        )
    return SchwarzTraceState(tuple(exchanges), sweep=sweep)


class SchwarzTraceQuantity(StrictModule):
    """One physical local quantity exchanged across every paired support."""

    left_operator: Any = eqx.field(static=True)
    right_operator: Any = eqx.field(static=True)
    quantity_id: str = eqx.field(static=True)

    def __init__(
        self,
        quantity_id: str,
        left_operator: Callable[[DomainFunction], object],
        right_operator: Callable[[DomainFunction], object],
        /,
    ) -> None:
        if not callable(left_operator) or not callable(right_operator):
            raise TypeError("Trace quantity operators must be callable.")
        identifier = str(quantity_id)
        if not identifier:
            raise ValueError("quantity_id must be non-empty.")
        self.quantity_id = identifier
        self.left_operator = left_operator
        self.right_operator = right_operator


class DiscreteOperatorTracePenalty(AbstractScalarTerm):
    """One-sided frozen-target fit for a generalized local trace quantity."""

    pairing: PairedSupport
    points: Any
    target: Array
    scale: Array
    operator: Any = eqx.field(static=True)
    fields: tuple[str, ...] = eqx.field(static=True)
    side: Literal["left", "right"] = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        pairing: PairedSupport,
        points: Any,
        target: Array,
        operator: Callable[[DomainFunction], object],
        /,
        *,
        side: Literal["left", "right"],
        scale: float = 1.0,
        label: str | None = None,
    ) -> None:
        if not callable(operator):
            raise TypeError("operator must be callable.")
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'.")
        scale_ = jnp.asarray(scale, dtype=jnp.float64).reshape(())
        if float(scale_) < 0.0:
            raise ValueError("scale must be non-negative.")
        self.fields = (str(field_name),)
        self.pairing = pairing
        self.points = points
        self.target = jnp.asarray(target)
        self.scale = scale_
        self.side = side
        self.label = None if label is None else str(label)
        self.operator = operator

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: PRNGKey = DOC_KEY0,
        iter_: int | Array | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_, kwargs
        quantity = self.operator(functions[self.fields[0]])
        if not isinstance(quantity, DomainFunction):
            raise TypeError("Generalized trace operator must return a DomainFunction.")
        trace = self.pairing.trace(quantity, side=self.side)(
            self.points,
            key=key,
        ).data
        residual = jnp.asarray(trace) - self.target
        return self.scale * jnp.mean(jnp.real(jnp.conj(residual) * residual))


def capture_generalized_trace_state(
    problem: FunctionalDecompositionProblem,
    functions: Mapping[str, DomainFunction],
    batches: Sequence[Any],
    quantity: SchwarzTraceQuantity,
    /,
    *,
    sweep: int,
    relaxation: float = 1.0,
    previous: SchwarzTraceState | None = None,
) -> SchwarzTraceState:
    """Capture a user-authored physical trace quantity on fixed paired supports."""
    if len(batches) != len(problem.cover.pairings):
        raise ValueError("One fixed trace batch is required per paired support.")
    if not isinstance(quantity, SchwarzTraceQuantity):
        raise TypeError("quantity must be a SchwarzTraceQuantity.")
    exchanges = []
    for pairing, points in zip(problem.cover.pairings, batches, strict=True):
        left_name = problem.family.field_name(pairing.left_patch_id)
        right_name = problem.family.field_name(pairing.right_patch_id)
        left_local = quantity.left_operator(functions[left_name])
        right_local = quantity.right_operator(functions[right_name])
        if not isinstance(left_local, DomainFunction) or not isinstance(
            right_local, DomainFunction
        ):
            raise TypeError("Trace quantity operators must return DomainFunction values.")
        left = pairing.trace(left_local, side="left")(points).data
        right = pairing.trace(right_local, side="right")(points).data
        if previous is None:
            left_target = right
            right_target = left
        else:
            old = previous.exchange(pairing.pairing_id)
            if old.quantity_id != quantity.quantity_id:
                raise ValueError("Generalized trace quantity identity changed.")
            left_target = (1.0 - relaxation) * old.left_target + relaxation * right
            right_target = (1.0 - relaxation) * old.right_target + relaxation * left
        exchanges.append(
            TraceExchangeState(
                pairing_id=pairing.pairing_id,
                points=points,
                left_values=left,
                right_values=right,
                left_target=left_target,
                right_target=right_target,
                quantity_id=quantity.quantity_id,
            )
        )
    return SchwarzTraceState(tuple(exchanges), sweep=sweep)


def _validate_aitken_trace_state(state: SchwarzTraceState, /) -> None:
    quantities = {exchange.quantity_id for exchange in state.exchanges}
    if len(quantities) != 1 or not all(quantities):
        raise ValueError(
            "Aitken Schwarz relaxation requires one homogeneous non-empty quantity_id."
        )


def _relax_aitken_sweep(
    previous: SchwarzTraceState,
    proposed: SchwarzTraceState,
    plan: AitkenRelaxation,
    previous_increment: Array | None,
    omega: Array,
    /,
) -> tuple[SchwarzTraceState, Array, Array]:
    """Relax one complete raw sweep without altering physical trace defects."""
    _validate_aitken_trace_state(previous)
    _validate_aitken_trace_state(proposed)
    if tuple(
        (exchange.pairing_id, exchange.quantity_id) for exchange in previous.exchanges
    ) != tuple(
        (exchange.pairing_id, exchange.quantity_id) for exchange in proposed.exchanges
    ):
        raise ValueError(
            "Aitken trace states must have matching pairings and quantities."
        )
    increments = tuple(
        (
            new.left_target - old.left_target,
            new.right_target - old.right_target,
        )
        for old, new in zip(previous.exchanges, proposed.exchanges, strict=True)
    )
    increment = jnp.concatenate(
        tuple(value.reshape((-1,)) for pair in increments for value in pair)
    )
    if previous_increment is not None:
        if previous_increment.shape != increment.shape:
            raise ValueError("Aitken previous increment must match the trace space.")
        omega = plan.update(previous_increment, increment, omega)
    exchanges = tuple(
        TraceExchangeState(
            pairing_id=new.pairing_id,
            points=new.points,
            left_values=new.left_values,
            right_values=new.right_values,
            left_target=old.left_target + omega * left_increment,
            right_target=old.right_target + omega * right_increment,
            quantity_id=new.quantity_id,
        )
        for old, new, (left_increment, right_increment) in zip(
            previous.exchanges, proposed.exchanges, increments, strict=True
        )
    )
    return SchwarzTraceState(exchanges, sweep=proposed.sweep), increment, omega


__all__ = [
    "DiscreteOperatorTracePenalty",
    "DiscreteTracePenalty",
    "SchwarzTraceState",
    "TraceExchangeState",
    "SchwarzTraceQuantity",
    "capture_generalized_trace_state",
    "capture_schwarz_trace_state",
]
