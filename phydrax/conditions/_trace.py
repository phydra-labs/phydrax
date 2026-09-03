#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx

from .._strict import StrictModule
from ._base import _validate_support, ConditionSupport


class AbstractJetDeclaration(StrictModule):
    """A symbolic field jet whose geometric realization is supplied later."""

    @property
    @abstractmethod
    def identity(self) -> tuple[Any, ...]:
        raise NotImplementedError

    def __mul__(self, coefficient: Any):
        return LinearTraceExpression(((coefficient, self),))

    def __rmul__(self, coefficient: Any):
        return self * coefficient

    def __add__(self, other: Any):
        return LinearTraceExpression(((1.0, self),)) + other

    def __radd__(self, other: Any):
        return self + other

    def __sub__(self, other: Any):
        return LinearTraceExpression(((1.0, self),)) - other


class FieldJet(AbstractJetDeclaration):
    """Legacy single-variable field jet declaration."""

    field: str = eqx.field(static=True)
    variable: str = eqx.field(static=True)
    order: int = eqx.field(static=True)
    normal: bool = eqx.field(static=True)

    def __init__(self, field: str, variable: str, order: int = 0, normal: bool = False):
        field_ = str(field)
        variable_ = str(variable)
        order_ = int(order)
        if not field_ or not variable_ or order_ < 0:
            raise ValueError("Field jets require names and nonnegative order.")
        if normal and order_ != 1:
            raise ValueError("Legacy normal field jets require order=1.")
        self.field = field_
        self.variable = variable_
        self.order = order_
        self.normal = bool(normal)

    @property
    def identity(self) -> tuple[Any, ...]:
        return ("field", self.field, self.variable, self.order, self.normal)


def _derivatives(
    value: Mapping[str, int] | Sequence[tuple[str, int]], /
) -> tuple[tuple[str, int], ...]:
    items = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    result = tuple((str(variable), int(order)) for variable, order in items)
    variables = tuple(variable for variable, _ in result)
    if any(not variable or order <= 0 for variable, order in result):
        raise ValueError(
            "Jet derivatives require non-empty variables and positive orders."
        )
    if len(set(variables)) != len(variables):
        raise ValueError("Each jet variable must occur exactly once.")
    return result


class PointJet(AbstractJetDeclaration):
    """A value or mixed derivative at a named point realization."""

    field: str = eqx.field(static=True)
    point_id: str = eqx.field(static=True)
    derivatives: tuple[tuple[str, int], ...] = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        point_id: str,
        /,
        *,
        derivatives: Mapping[str, int] | Sequence[tuple[str, int]] = (),
    ):
        field_ = str(field)
        point_ = str(point_id)
        if not field_ or not point_:
            raise ValueError("Point jets require non-empty field and point identities.")
        self.field = field_
        self.point_id = point_
        self.derivatives = _derivatives(derivatives)

    @property
    def identity(self) -> tuple[Any, ...]:
        return ("point", self.field, self.point_id, self.derivatives)


class TraceJet(AbstractJetDeclaration):
    """A mixed tangential/coordinate and normal derivative on a support."""

    field: str = eqx.field(static=True)
    trace_id: str = eqx.field(static=True)
    support: ConditionSupport
    derivatives: tuple[tuple[str, int], ...] = eqx.field(static=True)
    normal_order: int = eqx.field(static=True)
    side: Literal["interior", "exterior", "average", "jump"] = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        trace_id: str,
        support: ConditionSupport,
        /,
        *,
        derivatives: Mapping[str, int] | Sequence[tuple[str, int]] = (),
        normal_order: int = 0,
        side: Literal["interior", "exterior", "average", "jump"] = "interior",
    ):
        field_ = str(field)
        trace_ = str(trace_id)
        normal_ = int(normal_order)
        if not field_ or not trace_:
            raise ValueError("Trace jets require non-empty field and trace identities.")
        if normal_ < 0:
            raise ValueError("Trace normal_order must be nonnegative.")
        if side not in ("interior", "exterior", "average", "jump"):
            raise ValueError("Unknown TraceJet side.")
        self.field = field_
        self.trace_id = trace_
        self.support = _validate_support(support)
        self.derivatives = _derivatives(derivatives)
        self.normal_order = normal_
        self.side = side

    @property
    def identity(self) -> tuple[Any, ...]:
        return (
            "trace",
            self.field,
            self.trace_id,
            self.derivatives,
            self.normal_order,
            self.side,
        )


JetDeclaration: TypeAlias = FieldJet | PointJet | TraceJet


class LinearTraceExpression(StrictModule):
    terms: tuple[tuple[Any, JetDeclaration], ...]

    def __init__(self, terms: Sequence[tuple[Any, JetDeclaration]]):
        values = tuple(terms)
        if not values or any(
            not isinstance(jet, AbstractJetDeclaration) for _, jet in values
        ):
            raise TypeError("Linear trace expressions require declared jet terms.")
        self.terms = values

    def __add__(self, other: Any):
        expression = _trace_expression(other)
        return LinearTraceExpression((*self.terms, *expression.terms))

    def __radd__(self, other: Any):
        return self + other

    def __sub__(self, other: Any):
        expression = _trace_expression(other)
        return LinearTraceExpression(
            (*self.terms, *((-coefficient, jet) for coefficient, jet in expression.terms))
        )

    def __mul__(self, coefficient: Any):
        return LinearTraceExpression(
            tuple((coefficient * value, jet) for value, jet in self.terms)
        )

    def __rmul__(self, coefficient: Any):
        return self * coefficient


def _trace_expression(value: Any, /) -> LinearTraceExpression:
    if isinstance(value, LinearTraceExpression):
        return value
    if isinstance(value, AbstractJetDeclaration):
        return LinearTraceExpression(((1.0, value),))
    raise TypeError("Trace equations are linear combinations of declared jets.")


class LinearTraceEquation(StrictModule):
    lhs: LinearTraceExpression
    rhs: Any

    def __init__(self, lhs: JetDeclaration | LinearTraceExpression, rhs: Any):
        expression = _trace_expression(lhs)
        identities = tuple(jet.identity for _, jet in expression.terms)
        if len(set(identities)) != len(identities):
            raise ValueError("Trace equation terms must be canonical and unique.")
        self.lhs = expression
        self.rhs = rhs


def field_jet(
    field: str, variable: str, order: int = 0, normal: bool = False
) -> FieldJet:
    return FieldJet(field, variable, order, normal)


def point_jet(
    field: str,
    point_id: str,
    /,
    *,
    derivatives: Mapping[str, int] | Sequence[tuple[str, int]] = (),
) -> PointJet:
    return PointJet(field, point_id, derivatives=derivatives)


def trace_jet(
    field: str,
    trace_id: str,
    support: ConditionSupport,
    /,
    *,
    derivatives: Mapping[str, int] | Sequence[tuple[str, int]] = (),
    normal_order: int = 0,
    side: Literal["interior", "exterior", "average", "jump"] = "interior",
) -> TraceJet:
    return TraceJet(
        field,
        trace_id,
        support,
        derivatives=derivatives,
        normal_order=normal_order,
        side=side,
    )


def equal(lhs: JetDeclaration | LinearTraceExpression, rhs: Any) -> LinearTraceEquation:
    return LinearTraceEquation(lhs, rhs)


__all__ = [
    "AbstractJetDeclaration",
    "FieldJet",
    "JetDeclaration",
    "LinearTraceEquation",
    "LinearTraceExpression",
    "PointJet",
    "TraceJet",
    "equal",
    "field_jet",
    "point_jet",
    "trace_jet",
]
