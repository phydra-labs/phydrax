#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Mapping
from typing import Any, Literal

import equinox as eqx

from phydrax.domain import DomainComponent, DomainFunction, FixedStart

from ..operators.differential._domain_ops import dt_n
from ._base import _condition_functions, _fields, AbstractResidualCondition
from .boundary import _condition_value, ConditionValue


class Initial(AbstractResidualCondition):
    """Initial value or time-derivative condition on a fixed-start component."""

    fields: tuple[str, ...] = eqx.field(static=True)
    on: DomainComponent
    target: Any
    evolution_var: str = eqx.field(static=True)
    order: int = eqx.field(static=True)
    mode: Literal["reverse", "forward"] = eqx.field(static=True)
    backend: Literal["ad", "jet"] = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        on: DomainComponent,
        /,
        *,
        target: ConditionValue | None = None,
        evolution_var: str = "t",
        order: int = 0,
        mode: Literal["reverse", "forward"] = "reverse",
        backend: Literal["ad", "jet"] = "ad",
        label: str | None = None,
    ):
        if not isinstance(on, DomainComponent):
            raise TypeError("Initial conditions require one DomainComponent.")
        if evolution_var not in on.domain.labels:
            raise KeyError(f"Label {evolution_var!r} is not in domain {on.domain.labels}.")
        if not isinstance(on.spec.selection_for(evolution_var), FixedStart):
            raise ValueError("Initial conditions require FixedStart on the evolution variable.")
        resolved_order = int(order)
        if resolved_order < 0:
            raise ValueError("Initial derivative order must be nonnegative.")
        if mode not in ("reverse", "forward"):
            raise ValueError("mode must be 'reverse' or 'forward'.")
        if backend not in ("ad", "jet"):
            raise ValueError("backend must be 'ad' or 'jet'.")
        self.fields = _fields(field)
        self.on = on
        self.target = _condition_value(target, on, 0.0)
        self.evolution_var = str(evolution_var)
        self.order = resolved_order
        self.mode = mode
        self.backend = backend
        self.label = None if label is None else str(label)

    def residual(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> DomainFunction:
        (value,) = _condition_functions(self.fields, functions)
        derivative = dt_n(
            value,
            var=self.evolution_var,
            order=self.order,
            mode=self.mode,
            backend=self.backend,
        )
        return derivative - self.target


__all__ = ["Initial"]
