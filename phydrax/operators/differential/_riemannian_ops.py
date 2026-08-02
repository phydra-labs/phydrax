#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from string import ascii_lowercase
from typing import Any, Literal

import jax.numpy as jnp
import opt_einsum as oe

from ..._strict import StrictModule
from ...domain._base import _AbstractGeometry
from ...domain._function import DomainFunction
from ...metrix import LeviCivitaConnection, RiemannianMetric, TensorType
from ._domain_ops import _factor_and_dim, _resolve_var, grad, hessian


def _geometry_contract(
    function: DomainFunction,
    metric: RiemannianMetric,
    var: str | None,
    /,
) -> tuple[str, int]:
    var_ = _resolve_var(function, var)
    factor, dimension = _factor_and_dim(function, var_)
    if not isinstance(factor, _AbstractGeometry):
        raise ValueError("Riemannian operators require a geometry variable.")
    if dimension != metric.chart.dimension:
        raise ValueError(
            f"Metric chart dimension {metric.chart.dimension} does not match "
            f"domain variable {var_!r} dimension {dimension}."
        )
    return var_, dimension


def _dependencies(
    domain_labels: tuple[str, ...],
    functions: tuple[DomainFunction, ...],
    var: str,
    /,
) -> tuple[str, ...]:
    return tuple(
        label
        for label in domain_labels
        if label == var or any(label in function.deps for function in functions)
    )


def _positions(deps: tuple[str, ...], function: DomainFunction, /) -> tuple[int, ...]:
    index = {label: position for position, label in enumerate(deps)}
    return tuple(index[label] for label in function.deps)


class _RiemannianGradCallable(StrictModule):
    differential: DomainFunction
    metric: RiemannianMetric
    differential_positions: tuple[int, ...]
    coordinate_position: int

    def __init__(
        self,
        differential: DomainFunction,
        metric: RiemannianMetric,
        differential_positions: tuple[int, ...],
        coordinate_position: int,
        /,
    ):
        self.differential = differential
        self.metric = metric
        self.differential_positions = differential_positions
        self.coordinate_position = int(coordinate_position)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        differential = jnp.asarray(
            self.differential.func(
                *[args[position] for position in self.differential_positions],
                key=key,
                **kwargs,
            )
        )
        return oe.contract(
            "...ij,...j->...i",
            self.metric.inverse(args[self.coordinate_position]),
            differential,
        )


class _CovariantHessianCallable(StrictModule):
    differential: DomainFunction
    second_derivative: DomainFunction
    metric: RiemannianMetric
    differential_positions: tuple[int, ...]
    second_positions: tuple[int, ...]
    coordinate_position: int

    def __init__(
        self,
        differential: DomainFunction,
        second_derivative: DomainFunction,
        metric: RiemannianMetric,
        differential_positions: tuple[int, ...],
        second_positions: tuple[int, ...],
        coordinate_position: int,
        /,
    ):
        self.differential = differential
        self.second_derivative = second_derivative
        self.metric = metric
        self.differential_positions = differential_positions
        self.second_positions = second_positions
        self.coordinate_position = int(coordinate_position)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        differential = jnp.asarray(
            self.differential.func(
                *[args[position] for position in self.differential_positions],
                key=key,
                **kwargs,
            )
        )
        second = jnp.asarray(
            self.second_derivative.func(
                *[args[position] for position in self.second_positions],
                key=key,
                **kwargs,
            )
        )
        coefficients = LeviCivitaConnection(self.metric).coefficients(
            args[self.coordinate_position]
        )
        return second - oe.contract(
            "...kij,...k->...ij",
            coefficients,
            differential,
        )


class _RiemannianDivCallable(StrictModule):
    field: DomainFunction
    derivative: DomainFunction
    metric: RiemannianMetric
    field_positions: tuple[int, ...]
    derivative_positions: tuple[int, ...]
    coordinate_position: int
    dimension: int

    def __init__(
        self,
        field: DomainFunction,
        derivative: DomainFunction,
        metric: RiemannianMetric,
        field_positions: tuple[int, ...],
        derivative_positions: tuple[int, ...],
        coordinate_position: int,
        dimension: int,
        /,
    ):
        self.field = field
        self.derivative = derivative
        self.metric = metric
        self.field_positions = field_positions
        self.derivative_positions = derivative_positions
        self.coordinate_position = int(coordinate_position)
        self.dimension = int(dimension)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        values = jnp.asarray(
            self.field.func(
                *[args[position] for position in self.field_positions],
                key=key,
                **kwargs,
            )
        )
        if values.shape[-1:] != (self.dimension,):
            raise ValueError(
                "Riemannian divergence requires vector trailing dimension "
                f"{self.dimension}; got {values.shape}."
            )
        derivative = jnp.asarray(
            self.derivative.func(
                *[args[position] for position in self.derivative_positions],
                key=key,
                **kwargs,
            )
        )
        coefficients = LeviCivitaConnection(self.metric).coefficients(
            args[self.coordinate_position]
        )
        return jnp.trace(derivative, axis1=-2, axis2=-1) + oe.contract(
            "...iik,...k->...",
            coefficients,
            values,
        )


class _CovariantDerivativeCallable(StrictModule):
    field: DomainFunction
    derivative: DomainFunction
    metric: RiemannianMetric
    tensor_type: TensorType
    field_positions: tuple[int, ...]
    derivative_positions: tuple[int, ...]
    coordinate_position: int
    dimension: int

    def __init__(
        self,
        field: DomainFunction,
        derivative: DomainFunction,
        metric: RiemannianMetric,
        tensor_type: TensorType,
        field_positions: tuple[int, ...],
        derivative_positions: tuple[int, ...],
        coordinate_position: int,
        dimension: int,
        /,
    ):
        self.field = field
        self.derivative = derivative
        self.metric = metric
        self.tensor_type = tensor_type
        self.field_positions = field_positions
        self.derivative_positions = derivative_positions
        self.coordinate_position = int(coordinate_position)
        self.dimension = int(dimension)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        values = jnp.asarray(
            self.field.func(
                *[args[position] for position in self.field_positions],
                key=key,
                **kwargs,
            )
        )
        expected = (self.dimension,) * self.tensor_type.rank
        if values.shape[-self.tensor_type.rank :] != expected:
            raise ValueError(
                f"Tensor field must have trailing shape {expected}; got {values.shape}."
            )
        result = jnp.asarray(
            self.derivative.func(
                *[args[position] for position in self.derivative_positions],
                key=key,
                **kwargs,
            )
        )
        if self.tensor_type.rank == 0:
            return result
        coefficients = LeviCivitaConnection(self.metric).coefficients(
            args[self.coordinate_position]
        )
        letters = tuple(letter for letter in ascii_lowercase if letter not in ("x", "y"))[
            : self.tensor_type.rank
        ]
        output = "".join(letters) + "x"
        for slot, variance in enumerate(self.tensor_type.variance):
            input_letters = list(letters)
            input_letters[slot] = "y"
            if variance == "contravariant":
                connection_subscript = f"{letters[slot]}xy"
                sign = 1.0
            else:
                connection_subscript = f"yx{letters[slot]}"
                sign = -1.0
            correction = oe.contract(
                f"...{connection_subscript},...{''.join(input_letters)}->...{output}",
                coefficients,
                values,
            )
            result = result + sign * correction
        return result


class _RiemannianDivTensorCallable(StrictModule):
    derivative: DomainFunction
    derivative_positions: tuple[int, ...]
    dimension: int

    def __init__(
        self,
        derivative: DomainFunction,
        derivative_positions: tuple[int, ...],
        dimension: int,
        /,
    ):
        self.derivative = derivative
        self.derivative_positions = derivative_positions
        self.dimension = int(dimension)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        derivative = jnp.asarray(
            self.derivative.func(
                *[args[position] for position in self.derivative_positions],
                key=key,
                **kwargs,
            )
        )
        expected = (self.dimension, self.dimension, self.dimension)
        if derivative.shape[-3:] != expected:
            raise ValueError(
                "Riemannian tensor divergence requires derivative trailing shape "
                f"{expected}; got {derivative.shape}."
            )
        return jnp.trace(derivative, axis1=-2, axis2=-1)


class _LaplaceBeltramiCallable(StrictModule):
    hessian: DomainFunction
    metric: RiemannianMetric
    hessian_positions: tuple[int, ...]
    coordinate_position: int

    def __init__(
        self,
        hessian: DomainFunction,
        metric: RiemannianMetric,
        hessian_positions: tuple[int, ...],
        coordinate_position: int,
        /,
    ):
        self.hessian = hessian
        self.metric = metric
        self.hessian_positions = hessian_positions
        self.coordinate_position = int(coordinate_position)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        hessian = jnp.asarray(
            self.hessian.func(
                *[args[position] for position in self.hessian_positions],
                key=key,
                **kwargs,
            )
        )
        return oe.contract(
            "...ij,...ij->...",
            self.metric.inverse(args[self.coordinate_position]),
            hessian,
        )


def riemannian_grad(
    function: DomainFunction,
    metric: RiemannianMetric,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    var_, _ = _geometry_contract(function, metric, var)
    differential = grad(function, var=var_, mode=mode)
    if var_ not in function.deps:
        return differential
    deps = _dependencies(function.domain.labels, (differential,), var_)
    return DomainFunction(
        domain=function.domain,
        deps=deps,
        func=_RiemannianGradCallable(
            differential,
            metric,
            _positions(deps, differential),
            deps.index(var_),
        ),
        metadata=differential.metadata,
    )


def covariant_hessian(
    function: DomainFunction,
    metric: RiemannianMetric,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    var_, _ = _geometry_contract(function, metric, var)
    differential = grad(function, var=var_, mode=mode)
    second = hessian(function, var=var_)
    if var_ not in function.deps:
        return second
    deps = _dependencies(function.domain.labels, (differential, second), var_)
    return DomainFunction(
        domain=function.domain,
        deps=deps,
        func=_CovariantHessianCallable(
            differential,
            second,
            metric,
            _positions(deps, differential),
            _positions(deps, second),
            deps.index(var_),
        ),
        metadata=second.metadata,
    )


def riemannian_div(
    field: DomainFunction,
    metric: RiemannianMetric,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    var_, dimension = _geometry_contract(field, metric, var)
    derivative = grad(field, var=var_, mode=mode)
    deps = _dependencies(field.domain.labels, (field, derivative), var_)
    return DomainFunction(
        domain=field.domain,
        deps=deps,
        func=_RiemannianDivCallable(
            field,
            derivative,
            metric,
            _positions(deps, field),
            _positions(deps, derivative),
            deps.index(var_),
            dimension,
        ),
        metadata=derivative.metadata,
    )


def covariant_derivative(
    field: DomainFunction,
    metric: RiemannianMetric,
    tensor_type: TensorType,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    var_, dimension = _geometry_contract(field, metric, var)
    derivative = grad(field, var=var_, mode=mode)
    deps = _dependencies(field.domain.labels, (field, derivative), var_)
    return DomainFunction(
        domain=field.domain,
        deps=deps,
        func=_CovariantDerivativeCallable(
            field,
            derivative,
            metric,
            tensor_type,
            _positions(deps, field),
            _positions(deps, derivative),
            deps.index(var_),
            dimension,
        ),
        metadata=derivative.metadata,
    )


def riemannian_div_tensor(
    field: DomainFunction,
    metric: RiemannianMetric,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    """Covariant divergence ``∇_j T^ij`` of a rank-two contravariant field."""

    var_, dimension = _geometry_contract(field, metric, var)
    derivative = covariant_derivative(
        field,
        metric,
        TensorType(("contravariant", "contravariant")),
        var=var_,
        mode=mode,
    )
    deps = _dependencies(field.domain.labels, (derivative,), var_)
    return DomainFunction(
        domain=field.domain,
        deps=deps,
        func=_RiemannianDivTensorCallable(
            derivative,
            _positions(deps, derivative),
            dimension,
        ),
        metadata=derivative.metadata,
    )


def intrinsic_laplace_beltrami(
    function: DomainFunction,
    metric: RiemannianMetric,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    var_, _ = _geometry_contract(function, metric, var)
    covariant = covariant_hessian(function, metric, var=var_, mode=mode)
    if var_ not in function.deps:
        return DomainFunction(
            domain=function.domain,
            deps=function.deps,
            func=0.0,
            metadata=covariant.metadata,
        )
    deps = _dependencies(function.domain.labels, (covariant,), var_)
    return DomainFunction(
        domain=function.domain,
        deps=deps,
        func=_LaplaceBeltramiCallable(
            covariant,
            metric,
            _positions(deps, covariant),
            deps.index(var_),
        ),
        metadata=covariant.metadata,
    )
