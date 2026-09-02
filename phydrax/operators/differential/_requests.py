#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ...domain import (
    DerivativeBackend,
    DerivativeBasis,
    DerivativeMode,
    DerivativeRule,
    DomainFunction,
)


@dataclass(frozen=True, slots=True)
class DerivativeRequest:
    """One derivative of a named residual field requested by an operator."""

    field: str
    variable: str
    axes: tuple[int | None, ...]
    laplacian_count: int = 0

    @property
    def contracted_laplacian(self) -> bool:
        return self.laplacian_count > 0

    @property
    def order(self) -> int:
        return len(self.axes) + 2 * self.laplacian_count


class _RequestRecorderRule(DerivativeRule):
    def __init__(
        self,
        *,
        source: DomainFunction,
        field: str,
        requests: list[DerivativeRequest],
        prefix: tuple[int | None, ...] = (),
        prefix_laplacians: int = 0,
    ):
        self.source = source
        self.field = field
        self.requests = requests
        self.prefix = prefix
        self.prefix_laplacians = int(prefix_laplacians)

    def _result(
        self,
        *,
        prefix: tuple[int | None, ...],
        prefix_laplacians: int,
    ) -> DomainFunction:
        return DomainFunction(
            domain=self.source.domain,
            deps=self.source.deps,
            func=self.source.func,
            metadata=self.source.metadata,
            derivative_rule=_RequestRecorderRule(
                source=self.source,
                field=self.field,
                requests=self.requests,
                prefix=prefix,
                prefix_laplacians=prefix_laplacians,
            ),
        )

    def derive(
        self,
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        del mode, backend, basis, periodic
        axes = self.prefix + (axis,) * int(order)
        request = DerivativeRequest(
            field=self.field,
            variable=var,
            axes=axes,
            laplacian_count=self.prefix_laplacians,
        )
        self.requests.append(request)
        return self._result(
            prefix=axes,
            prefix_laplacians=self.prefix_laplacians,
        )

    def derive_laplacian(
        self,
        *,
        var: str,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        del mode, backend, basis, periodic
        request = DerivativeRequest(
            field=self.field,
            variable=var,
            axes=self.prefix,
            laplacian_count=self.prefix_laplacians + 1,
        )
        self.requests.append(request)
        return self._result(
            prefix=self.prefix,
            prefix_laplacians=self.prefix_laplacians + 1,
        )


def trace_derivative_requests(
    residual: Callable[[Mapping[str, DomainFunction]], DomainFunction],
    functions: Mapping[str, DomainFunction],
    /,
) -> tuple[DerivativeRequest, ...]:
    """Trace derivative requirements without evaluating a batch."""

    recorded: list[DerivativeRequest] = []
    traced = {
        name: function.with_derivative_rule(
            _RequestRecorderRule(
                source=function,
                field=name,
                requests=recorded,
            )
        )
        for name, function in functions.items()
    }
    result = residual(traced)
    if not isinstance(result, DomainFunction):
        raise TypeError("A ResidualPenalty condition must return a DomainFunction.")

    unique: list[DerivativeRequest] = []
    seen: set[DerivativeRequest] = set()
    for request in recorded:
        if request not in seen:
            seen.add(request)
            unique.append(request)
    return tuple(unique)



DerivativeExecutionStrategy = Literal["reverse", "forward", "jvp", "jet"]


class DerivativeExecutionPlan(StrictModule):
    """Static execution recommendation for one traced residual derivative set."""

    requests: tuple[DerivativeRequest, ...] = eqx.field(static=True)
    strategy: DerivativeExecutionStrategy = eqx.field(static=True)
    maximum_order: int = eqx.field(static=True)
    variable_count: int = eqx.field(static=True)
    contracted_laplacian: bool = eqx.field(static=True)

    def __init__(
        self,
        requests: tuple[DerivativeRequest, ...],
        strategy: DerivativeExecutionStrategy,
        /,
    ):
        if not requests:
            raise ValueError("DerivativeExecutionPlan requires derivative requests.")
        if strategy not in ("reverse", "forward", "jvp", "jet"):
            raise ValueError("Unknown derivative execution strategy.")
        self.requests = tuple(requests)
        self.strategy = strategy
        self.maximum_order = max(request.order for request in requests)
        self.variable_count = len({request.variable for request in requests})
        self.contracted_laplacian = any(
            request.contracted_laplacian for request in requests
        )


def plan_derivative_execution(
    requests: tuple[DerivativeRequest, ...],
    /,
    *,
    output_size: int | None = None,
    coordinate_size: int | None = None,
) -> DerivativeExecutionPlan:
    """Choose a non-approximating AD strategy from derivative request shape."""
    values = tuple(requests)
    if not values:
        raise ValueError("At least one derivative request is required.")
    maximum_order = max(request.order for request in values)
    contracted = any(request.contracted_laplacian for request in values)
    if maximum_order > 2:
        strategy: DerivativeExecutionStrategy = "jet"
    elif contracted or maximum_order == 2:
        strategy = "jvp"
    elif (
        output_size is not None
        and coordinate_size is not None
        and int(output_size) > int(coordinate_size)
    ):
        strategy = "forward"
    else:
        strategy = "reverse"
    return DerivativeExecutionPlan(values, strategy)


class FusedDerivativeEvaluation(StrictModule):
    value: Any
    first_derivatives: tuple[Any, ...]
    diagonal_second_derivatives: tuple[Any, ...]
    first_axes: tuple[int, ...] = eqx.field(static=True)
    second_axes: tuple[int, ...] = eqx.field(static=True)


def evaluate_fused_coordinate_derivatives(
    function: Callable[[Array], Any],
    point: Array,
    /,
    *,
    first_axes: tuple[int, ...] = (),
    second_axes: tuple[int, ...] = (),
) -> FusedDerivativeEvaluation:
    """Evaluate a value, Jacobian columns, and requested Hessian diagonal entries."""
    if not callable(function):
        raise TypeError("function must be callable.")
    point_ = jnp.asarray(point)
    if point_.ndim != 1:
        raise ValueError("Fused coordinate derivatives require one rank-one point.")
    first = tuple(int(axis) for axis in first_axes)
    second = tuple(int(axis) for axis in second_axes)
    if any(axis < 0 or axis >= point_.size for axis in first + second):
        raise ValueError("Fused derivative axis is out of range.")
    value, pushforward = jax.linearize(function, point_)

    def direction(axis: int, /) -> Array:
        return jnp.zeros_like(point_).at[axis].set(1.0)

    first_values = tuple(pushforward(direction(axis)) for axis in first)
    second_values = []
    for axis in second:
        tangent = direction(axis)

        def first_direction(current, _tangent=tangent):
            return jax.jvp(function, (current,), (_tangent,))[1]

        second_values.append(
            jax.jvp(first_direction, (point_,), (tangent,))[1]
        )
    return FusedDerivativeEvaluation(
        value,
        first_values,
        tuple(second_values),
        first,
        second,
    )
__all__ = [
    "DerivativeExecutionPlan",
    "DerivativeExecutionStrategy",
    "DerivativeRequest",
    "FusedDerivativeEvaluation",
    "evaluate_fused_coordinate_derivatives",
    "plan_derivative_execution",
    "trace_derivative_requests",
]
