#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from ..._term import AbstractScalarTerm
from ...domain import DomainFunction, PairedSupport
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
    ):
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
    ):
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
    ):
        if not isinstance(pairing, PairedSupport):
            raise TypeError("pairing must be a PairedSupport.")
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'.")
        target_ = jnp.asarray(target)
        scale_ = jnp.asarray(scale, dtype=float).reshape(())
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
        key: Key[Array, ""] = DOC_KEY0,
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
        left_operator,
        right_operator,
        /,
    ):
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
        operator,
        /,
        *,
        side: Literal["left", "right"],
        scale: float = 1.0,
        label: str | None = None,
    ):
        if not callable(operator):
            raise TypeError("operator must be callable.")
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'.")
        scale_ = jnp.asarray(scale, dtype=float).reshape(())
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
        key: Key[Array, ""] = DOC_KEY0,
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


class AitkenTracePlan(StrictModule):
    initial_relaxation: float = eqx.field(static=True)
    minimum_relaxation: float = eqx.field(static=True)
    maximum_relaxation: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_relaxation: float = 1.0,
        minimum_relaxation: float = 0.05,
        maximum_relaxation: float = 1.5,
    ):
        initial = float(initial_relaxation)
        minimum = float(minimum_relaxation)
        maximum = float(maximum_relaxation)
        if not 0.0 < minimum <= initial <= maximum:
            raise ValueError("Aitken relaxation bounds are invalid.")
        self.initial_relaxation = initial
        self.minimum_relaxation = minimum
        self.maximum_relaxation = maximum


class AitkenTraceState(StrictModule):
    residual: Array
    relaxation: Array

    def __init__(self, residual: Array, relaxation: Array, /):
        self.residual = jnp.asarray(residual)
        self.relaxation = jnp.asarray(relaxation).reshape(())


def aitken_relax_trace_state(
    previous: SchwarzTraceState,
    proposed: SchwarzTraceState,
    plan: AitkenTracePlan | None = None,
    state: AitkenTraceState | None = None,
    /,
) -> tuple[SchwarzTraceState, AitkenTraceState]:
    """Apply bounded scalar Aitken delta-squared relaxation in trace space."""
    plan_ = AitkenTracePlan() if plan is None else plan
    if not isinstance(plan_, AitkenTracePlan):
        raise TypeError("plan must be an AitkenTracePlan or None.")
    if len(previous.exchanges) != len(proposed.exchanges):
        raise ValueError("Aitken trace states must have matching pairings.")
    residual = jnp.concatenate(
        tuple(
            (new.left_target - old.left_target).reshape((-1,))
            for old, new in zip(
                previous.exchanges,
                proposed.exchanges,
                strict=True,
            )
        )
        + tuple(
            (new.right_target - old.right_target).reshape((-1,))
            for old, new in zip(
                previous.exchanges,
                proposed.exchanges,
                strict=True,
            )
        )
    )
    if state is None:
        relaxation = jnp.asarray(plan_.initial_relaxation)
    else:
        delta = residual - state.residual
        denominator = jnp.real(jnp.vdot(delta, delta))
        numerator = jnp.real(jnp.vdot(state.residual, delta))
        candidate = (
            -state.relaxation
            * numerator
            / jnp.where(
                denominator > 0.0,
                denominator,
                1.0,
            )
        )
        relaxation = jnp.clip(
            jnp.where(denominator > 0.0, candidate, state.relaxation),
            plan_.minimum_relaxation,
            plan_.maximum_relaxation,
        )
    exchanges = tuple(
        TraceExchangeState(
            pairing_id=new.pairing_id,
            points=new.points,
            left_values=new.left_values,
            right_values=new.right_values,
            left_target=old.left_target
            + relaxation * (new.left_target - old.left_target),
            right_target=old.right_target
            + relaxation * (new.right_target - old.right_target),
            quantity_id=new.quantity_id,
        )
        for old, new in zip(previous.exchanges, proposed.exchanges, strict=True)
    )
    return (
        SchwarzTraceState(exchanges, sweep=proposed.sweep),
        AitkenTraceState(residual, relaxation),
    )


__all__ = [
    "AitkenTracePlan",
    "AitkenTraceState",
    "DiscreteOperatorTracePenalty",
    "DiscreteTracePenalty",
    "SchwarzTraceState",
    "TraceExchangeState",
    "SchwarzTraceQuantity",
    "aitken_relax_trace_state",
    "capture_generalized_trace_state",
    "capture_schwarz_trace_state",
]
