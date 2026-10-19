#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Bounded, coefficient-independent preparation of native interval derivative programs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from math import factorial, prod
from typing import cast, Protocol

import jax
import numpy as np
from jax import Array
from jax.experimental.jet import jet
from jax.extend import core as jax_core
from jax.tree_util import PyTreeDef

from ..._strict import StrictModule
from .._interval_enclosure import (
    coefficient_interval,
    prepare_interval_function,
    PreparedIntervalFunction,
)


class _Evaluator(Protocol):
    def evaluate(self, parameters: Array, /) -> Array: ...


def _static_signature(value: object, /) -> bool:
    if value is None or isinstance(value, (bool, int, type)):
        return True
    if isinstance(value, tuple):
        return all(_static_signature(item) for item in value)
    return False


def _module_auxiliary_signature(auxiliary: object, /) -> bool:
    # Equinox's first auxiliary slot is its wrapper sentinel, or a
    # wrapper-data tuple. The latter may retain a source instance.
    if (
        not isinstance(auxiliary, tuple)
        or not auxiliary
        or isinstance(auxiliary[0], tuple)
    ):
        return False
    return all(_static_signature(value) for value in auxiliary[1:])


def _data_free_structure(structure: PyTreeDef, /) -> bool:
    node = structure.node_data()
    if node is not None:
        kind, auxiliary = node
        match kind:
            case type() if issubclass(kind, StrictModule):
                if not _module_auxiliary_signature(auxiliary):
                    return False
            case _:
                if auxiliary is not None or kind not in (tuple, list, type(None)):
                    return False
    return all(_data_free_structure(child) for child in structure.children())


def _program_has_arrays(value: object, /) -> bool:
    pending = [value]
    visited: set[int] = set()
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in visited:
            continue
        visited.add(identity)
        match item:
            case Array() | np.ndarray():
                return True
            case jax_core.ClosedJaxpr():
                if item.consts:
                    return True
                # Current JAX exposes a unified jaxpr whose .jaxpr can be self.
                pending.extend(equation.params for equation in item.jaxpr.eqns)
            case jax_core.Jaxpr():
                pending.extend(equation.params for equation in item.eqns)
            case Mapping():
                pending.extend(item.values())
            case tuple() | list():
                pending.extend(item)
    return False


@lru_cache(maxsize=64)
def _derivative_template(
    structure: PyTreeDef,
    shapes: tuple[tuple[int, ...], ...],
    dtypes: tuple[str, ...],
    implementation: Callable[..., Array],
    parameter_dimension: int,
    order: int,
    batch_capacity: int,
) -> PreparedIntervalFunction | None:
    dimension = parameter_dimension + sum(prod(shape) for shape in shapes)

    def evaluate(values: Array) -> Array:
        offset = parameter_dimension
        leaves: list[Array] = []
        for shape, dtype in zip(shapes, dtypes, strict=True):
            count = prod(shape)
            leaves.append(
                values[offset : offset + count].reshape(shape).astype(np.dtype(dtype))
            )
            offset += count
        # The admitted PyTree definition fixes the registered source type and
        # declared field set; reconstruction remains with the native framework.
        evaluator = cast(_Evaluator, jax.tree_util.tree_unflatten(structure, leaves))

        def point(parameters: Array) -> Array:
            return implementation(evaluator, parameters)

        derivative = point
        for _ in range(order):
            derivative = jax.jacfwd(derivative)
        parameters = values[:parameter_dimension]
        return derivative(parameters[0] if parameter_dimension == 1 else parameters)

    prepared = prepare_interval_function(
        evaluate, dimension, batch_capacity=batch_capacity
    )
    if _program_has_arrays(prepared.closed_jaxpr):
        return None
    return prepared


@dataclass(frozen=True, slots=True)
class _BoundDerivativeTemplate:
    program: PreparedIntervalFunction
    coefficient_lower: np.ndarray
    coefficient_upper: np.ndarray

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        count = lower.shape[0]
        return self.program.evaluate(
            np.concatenate(
                (
                    lower,
                    np.broadcast_to(
                        self.coefficient_lower, (count, self.coefficient_lower.size)
                    ),
                ),
                axis=1,
            ),
            np.concatenate(
                (
                    upper,
                    np.broadcast_to(
                        self.coefficient_upper, (count, self.coefficient_upper.size)
                    ),
                ),
                axis=1,
            ),
        )


def _bind_coefficients(
    program: PreparedIntervalFunction,
    arrays: tuple[np.ndarray, ...],
    constant_bounds: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
    /,
) -> _BoundDerivativeTemplate:
    bounds = tuple(coefficient_interval(value, constant_bounds) for value in arrays)
    return _BoundDerivativeTemplate(
        program,
        np.concatenate(
            tuple(np.asarray(lower, dtype=np.float64).reshape(-1) for lower, _ in bounds)
        ),
        np.concatenate(
            tuple(np.asarray(upper, dtype=np.float64).reshape(-1) for _, upper in bounds)
        ),
    )


def prepare_derivative_interval(
    evaluator: _Evaluator,
    parameter_dimension: int,
    order: int,
    /,
    *,
    batch_capacity: int,
    constant_bounds: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
) -> PreparedIntervalFunction | _BoundDerivativeTemplate:
    leaves, structure = jax.tree_util.tree_flatten(evaluator)
    if _data_free_structure(structure) and all(
        isinstance(value, (Array, np.ndarray)) for value in leaves
    ):
        arrays = tuple(np.asarray(value) for value in leaves)
        program = _derivative_template(
            structure,
            tuple(value.shape for value in arrays),
            tuple(value.dtype.str for value in arrays),
            type(evaluator).evaluate,
            parameter_dimension,
            order,
            batch_capacity,
        )
        if program is not None:
            return _bind_coefficients(program, arrays, constant_bounds)
    derivative = evaluator.evaluate
    for _ in range(order):
        derivative = jax.jacfwd(derivative)
    if parameter_dimension == 1:

        def curve_derivative(parameters: Array) -> Array:
            return derivative(parameters[0])

        function = curve_derivative
    else:
        function = derivative
    return prepare_interval_function(
        function,
        parameter_dimension,
        batch_capacity=batch_capacity,
        constant_bounds=constant_bounds,
    )


def prepare_taylor_derivative_interval(
    evaluator: _Evaluator,
    parameter_dimension: int,
    axis: int,
    order: int,
    /,
    *,
    batch_capacity: int,
    constant_bounds: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
) -> PreparedIntervalFunction:
    """Enclose a pure directional derivative without materializing a tensor jet.

    Taylor coefficients are factorial-unscaled during propagation, so the final
    conversion uses an exact integer rather than the transform's gamma function.
    Programs stay request-local: their source coefficients must not enter a
    global compilation cache.
    """
    if not 0 <= axis < parameter_dimension or order < 1:
        raise ValueError(
            "TaylorDerivative.parameters must name a parameter axis and positive order; "
            f"got {(axis, order)!r} for dimension {parameter_dimension}."
        )
    direction = np.zeros(parameter_dimension, dtype=np.float64)
    direction[axis] = 1

    def derivative(parameters: Array) -> Array:
        tangent = jax.numpy.asarray(direction)
        zeros = jax.numpy.zeros_like(parameters)
        _, coefficients = jet(
            evaluator.evaluate,
            (parameters,),
            ((tangent,) + (zeros,) * (order - 1),),
            factorial_scaled=False,
        )
        return coefficients[-1] * factorial(order)

    return prepare_interval_function(
        derivative,
        parameter_dimension,
        batch_capacity=batch_capacity,
        constant_bounds=constant_bounds,
    )
