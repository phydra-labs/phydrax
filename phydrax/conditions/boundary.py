#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Mapping
from typing import Any, Literal

import equinox as eqx
from jaxtyping import ArrayLike

from phydrax.domain import (
    DomainComponent,
    DomainFunction,
    Fixed,
    FixedEnd,
    FixedStart,
)

from ..operators.differential._domain_ops import directional_derivative, dt
from ._base import (
    _condition_functions,
    _fields,
    _validate_support,
    AbstractResidualCondition,
    ConditionSupport,
)


ConditionValue = DomainFunction | ArrayLike | Callable[..., Any]


def _non_fixed_labels(component: DomainComponent, /) -> tuple[str, ...]:
    fixed = {
        label
        for label in component.domain.labels
        if isinstance(component.spec.selection_for(label), (FixedStart, FixedEnd, Fixed))
    }
    return tuple(label for label in component.domain.labels if label not in fixed)


def _value_deps(component: ConditionSupport, /) -> tuple[str, ...]:
    if isinstance(component, DomainComponent):
        return _non_fixed_labels(component)
    return component.domain.labels


def _condition_value(value: ConditionValue | None, on: ConditionSupport, default: float):
    if value is None:
        return default
    if isinstance(value, DomainFunction):
        if not value.domain.same_support(on.domain):
            raise ValueError("Condition target domain is incompatible with support.")
        return value
    if callable(value):
        return on.domain.Function(*_value_deps(on))(value)
    return value


class Dirichlet(AbstractResidualCondition):
    """Value condition ``field = target`` on a domain component."""

    fields: tuple[str, ...] = eqx.field(static=True)
    on: ConditionSupport
    target: Any
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        on: ConditionSupport,
        /,
        *,
        target: ConditionValue | None = None,
        label: str | None = None,
    ):
        self.fields = _fields(field)
        self.on = _validate_support(on)
        self.target = _condition_value(target, self.on, 0.0)
        self.label = None if label is None else str(label)

    def residual(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> DomainFunction:
        (value,) = _condition_functions(self.fields, functions)
        return value - self.target


class Neumann(AbstractResidualCondition):
    """Outward normal derivative condition on a boundary component."""

    fields: tuple[str, ...] = eqx.field(static=True)
    on: DomainComponent
    target: Any
    normal: DomainFunction
    var: str = eqx.field(static=True)
    mode: Literal["reverse", "forward"] = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        on: DomainComponent,
        /,
        *,
        var: str = "x",
        target: ConditionValue | None = None,
        mode: Literal["reverse", "forward"] = "reverse",
        label: str | None = None,
    ):
        if not isinstance(on, DomainComponent):
            raise TypeError("Neumann conditions require one DomainComponent.")
        if mode not in ("reverse", "forward"):
            raise ValueError("mode must be 'reverse' or 'forward'.")
        self.fields = _fields(field)
        self.on = on
        self.target = _condition_value(target, on, 0.0)
        self.normal = on.normal(var=var)
        self.var = str(var)
        self.mode = mode
        self.label = None if label is None else str(label)

    def residual(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> DomainFunction:
        (value,) = _condition_functions(self.fields, functions)
        derivative = directional_derivative(
            value,
            self.normal,
            var=self.var,
            mode=self.mode,
        )
        return derivative - self.target


class Robin(AbstractResidualCondition):
    """Linear value/normal-derivative condition on a boundary component."""

    fields: tuple[str, ...] = eqx.field(static=True)
    on: DomainComponent
    dirichlet_coefficient: Any
    neumann_coefficient: Any
    target: Any
    normal: DomainFunction
    var: str = eqx.field(static=True)
    mode: Literal["reverse", "forward"] = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        on: DomainComponent,
        /,
        *,
        dirichlet_coefficient: ConditionValue | None = None,
        neumann_coefficient: ConditionValue | None = None,
        target: ConditionValue | None = None,
        var: str = "x",
        mode: Literal["reverse", "forward"] = "reverse",
        label: str | None = None,
    ):
        if not isinstance(on, DomainComponent):
            raise TypeError("Robin conditions require one DomainComponent.")
        if mode not in ("reverse", "forward"):
            raise ValueError("mode must be 'reverse' or 'forward'.")
        self.fields = _fields(field)
        self.on = on
        self.dirichlet_coefficient = _condition_value(dirichlet_coefficient, on, 0.0)
        self.neumann_coefficient = _condition_value(neumann_coefficient, on, 0.0)
        self.target = _condition_value(target, on, 0.0)
        self.normal = on.normal(var=var)
        self.var = str(var)
        self.mode = mode
        self.label = None if label is None else str(label)

    def residual(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> DomainFunction:
        (value,) = _condition_functions(self.fields, functions)
        derivative = directional_derivative(
            value,
            self.normal,
            var=self.var,
            mode=self.mode,
        )
        return (
            self.dirichlet_coefficient * value
            + self.neumann_coefficient * derivative
            - self.target
        )


class Absorbing(AbstractResidualCondition):
    """First-order absorbing/Sommerfeld boundary condition."""

    fields: tuple[str, ...] = eqx.field(static=True)
    on: DomainComponent
    wavespeed: Any
    target: Any
    normal: DomainFunction
    var: str = eqx.field(static=True)
    time_var: str = eqx.field(static=True)
    mode: Literal["reverse", "forward"] = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        on: DomainComponent,
        /,
        *,
        var: str = "x",
        time_var: str = "t",
        wavespeed: ConditionValue | None = None,
        target: ConditionValue | None = None,
        mode: Literal["reverse", "forward"] = "reverse",
        label: str | None = None,
    ):
        if not isinstance(on, DomainComponent):
            raise TypeError("Absorbing conditions require one DomainComponent.")
        if time_var not in on.domain.labels:
            raise KeyError(f"Label {time_var!r} is not in domain {on.domain.labels}.")
        if mode not in ("reverse", "forward"):
            raise ValueError("mode must be 'reverse' or 'forward'.")
        self.fields = _fields(field)
        self.on = on
        self.wavespeed = _condition_value(wavespeed, on, 1.0)
        self.target = _condition_value(target, on, 0.0)
        self.normal = on.normal(var=var)
        self.var = str(var)
        self.time_var = str(time_var)
        self.mode = mode
        self.label = None if label is None else str(label)

    def residual(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> DomainFunction:
        (value,) = _condition_functions(self.fields, functions)
        normal_derivative = directional_derivative(
            value,
            self.normal,
            var=self.var,
            mode=self.mode,
        )
        time_derivative = dt(value, var=self.time_var, mode=self.mode)
        return normal_derivative + time_derivative / self.wavespeed - self.target


__all__ = ["Absorbing", "ConditionValue", "Dirichlet", "Neumann", "Robin"]
