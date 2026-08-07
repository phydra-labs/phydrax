#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
import operator as py_operator
from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp

from phydrax.domain import DomainFunction

from ..._strict import StrictModule
from ._validation import join_function_arguments, validate_matrix_value


def _subsystem_dimensions(subsystem_dims: Sequence[int], /) -> tuple[int, ...]:
    if isinstance(subsystem_dims, (str, bytes)) or not isinstance(
        subsystem_dims, Sequence
    ):
        raise TypeError("subsystem_dims must be a sequence of positive integers.")
    if not subsystem_dims:
        raise ValueError("subsystem_dims must contain at least one dimension.")
    dimensions = []
    for position, value in enumerate(subsystem_dims):
        if isinstance(value, bool):
            raise TypeError(
                "subsystem_dims must contain positive integers; "
                f"item {position} is bool."
            )
        try:
            dimension = py_operator.index(value)
        except TypeError as exc:
            raise TypeError(
                "subsystem_dims must contain positive integers; "
                f"item {position} is {type(value).__name__}."
            ) from exc
        if dimension <= 0:
            raise ValueError(
                "subsystem_dims must contain positive integers; "
                f"item {position} is {dimension}."
            )
        dimensions.append(dimension)
    return tuple(dimensions)


def _subsystem_index(value: Any, /, *, count: int, role: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{role} must be an integer, got bool.")
    try:
        index = py_operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{role} must be an integer, got {type(value).__name__}.") from exc
    if index < 0 or index >= count:
        raise ValueError(f"{role} must be in [0, {count}), got {index}.")
    return index


def _traced_subsystems(
    trace_out: int | Sequence[int],
    /,
    *,
    count: int,
) -> tuple[int, ...]:
    if isinstance(trace_out, Sequence) and not isinstance(trace_out, (str, bytes)):
        values = tuple(
            _subsystem_index(value, count=count, role=f"trace_out item {position}")
            for position, value in enumerate(trace_out)
        )
    else:
        values = (_subsystem_index(trace_out, count=count, role="trace_out"),)
    if len(set(values)) != len(values):
        raise ValueError("trace_out subsystem indices must be unique.")
    return tuple(sorted(values))


class _TensorProductCallable(StrictModule):
    factors: tuple[DomainFunction, ...]
    factor_positions: tuple[tuple[int, ...], ...]

    def __init__(
        self,
        factors: tuple[DomainFunction, ...],
        factor_positions: tuple[tuple[int, ...], ...],
    ):
        self.factors = factors
        self.factor_positions = factor_positions

    def __call__(self, *args, key=None, **kwargs):
        result = None
        value_rank = None
        for index, (factor, positions) in enumerate(
            zip(self.factors, self.factor_positions, strict=True)
        ):
            factor_args = tuple(args[position] for position in positions)
            value = jnp.asarray(factor.func(*factor_args, key=key, **kwargs))
            if value.ndim not in (1, 2):
                raise ValueError(
                    "tensor_product factors must be vectors or square matrices; "
                    f"factor {index} has shape {value.shape}."
                )
            if int(value.shape[0]) == 0 or (
                value.ndim == 2 and int(value.shape[1]) != int(value.shape[0])
            ):
                raise ValueError(
                    "tensor_product factors must be nonempty vectors or square "
                    f"matrices; factor {index} has shape {value.shape}."
                )
            if value_rank is None:
                value_rank = value.ndim
            elif value.ndim != value_rank:
                raise ValueError(
                    "tensor_product factors must all be vector-valued or all be "
                    "matrix-valued; "
                    f"factor 0 has rank {value_rank}, factor {index} has rank "
                    f"{value.ndim}."
                )
            result = value if result is None else jnp.kron(result, value)
        if result is None:
            raise RuntimeError("tensor_product received no factors after validation.")
        return result


class _PartialTraceCallable(StrictModule):
    density: DomainFunction
    subsystem_dims: tuple[int, ...]
    trace_out: tuple[int, ...]
    total_dimension: int

    def __init__(
        self,
        density: DomainFunction,
        subsystem_dims: tuple[int, ...],
        trace_out: tuple[int, ...],
    ):
        self.density = density
        self.subsystem_dims = subsystem_dims
        self.trace_out = trace_out
        self.total_dimension = math.prod(subsystem_dims)

    def __call__(self, *args, key=None, **kwargs):
        density = validate_matrix_value(
            self.density.func(*args, key=key, **kwargs),
            role="partial_trace density operator",
        )
        if int(density.shape[0]) != self.total_dimension:
            raise ValueError(
                "Density dimension must equal the product of subsystem_dims; "
                f"got density shape {density.shape} and subsystem_dims "
                f"{self.subsystem_dims} (product {self.total_dimension})."
            )
        result = jnp.reshape(density, self.subsystem_dims + self.subsystem_dims)
        remaining_dims = list(self.subsystem_dims)
        subsystem_count = len(remaining_dims)
        for subsystem in reversed(self.trace_out):
            result = jnp.trace(
                result,
                axis1=subsystem,
                axis2=subsystem + subsystem_count,
            )
            remaining_dims.pop(subsystem)
            subsystem_count -= 1
        if not remaining_dims:
            return jnp.reshape(result, ())
        reduced_dimension = math.prod(remaining_dims)
        return jnp.reshape(result, (reduced_dimension, reduced_dimension))


class _EmbeddedOperatorCallable(StrictModule):
    source: DomainFunction
    subsystem_dims: tuple[int, ...]
    subsystem: int
    left_dimension: int
    right_dimension: int

    def __init__(
        self,
        source: DomainFunction,
        subsystem_dims: tuple[int, ...],
        subsystem: int,
    ):
        self.source = source
        self.subsystem_dims = subsystem_dims
        self.subsystem = subsystem
        self.left_dimension = math.prod(subsystem_dims[:subsystem])
        self.right_dimension = math.prod(subsystem_dims[subsystem + 1 :])

    def __call__(self, *args, key=None, **kwargs):
        source = validate_matrix_value(
            self.source.func(*args, key=key, **kwargs),
            role="embedded subsystem operator",
        )
        expected = self.subsystem_dims[self.subsystem]
        if int(source.shape[0]) != expected:
            raise ValueError(
                "Operator dimension must match the selected subsystem; "
                f"got operator shape {source.shape}, subsystem {self.subsystem} "
                f"has dimension {expected}."
            )
        result = source
        if self.left_dimension != 1:
            result = jnp.kron(
                jnp.eye(self.left_dimension, dtype=source.dtype),
                result,
            )
        if self.right_dimension != 1:
            result = jnp.kron(
                result,
                jnp.eye(self.right_dimension, dtype=source.dtype),
            )
        return result


def tensor_product(*factors: DomainFunction) -> DomainFunction:
    r"""Construct the pointwise Kronecker product of quantum factors.

    Every factor must be a ``DomainFunction``. At evaluation, their values must all be
    vectors or all be square matrices; mixed vector/matrix products are rejected to
    keep the output contract unambiguous.
    """
    if not factors:
        raise ValueError("tensor_product requires at least one DomainFunction.")
    for index, factor in enumerate(factors):
        if not isinstance(factor, DomainFunction):
            raise TypeError(
                "tensor_product expects only DomainFunctions; "
                f"factor {index} is {type(factor).__name__}."
            )
    domain, deps, promoted, positions = join_function_arguments(*factors)
    return DomainFunction(
        domain=domain,
        deps=deps,
        func=_TensorProductCallable(promoted, positions),
        metadata={},
    )


def partial_trace(
    density: DomainFunction,
    /,
    *,
    subsystem_dims: Sequence[int],
    trace_out: int | Sequence[int],
) -> DomainFunction:
    r"""Trace selected subsystems out of a composite density operator.

    ``subsystem_dims`` explicitly defines the Hilbert-space factorization. Untraced
    subsystems remain in their original order. Tracing every subsystem returns the
    scalar total trace; tracing none returns the original matrix value.
    """
    if not isinstance(density, DomainFunction):
        raise TypeError("partial_trace density must be a DomainFunction.")
    dimensions = _subsystem_dimensions(subsystem_dims)
    traced = _traced_subsystems(trace_out, count=len(dimensions))
    return DomainFunction(
        domain=density.domain,
        deps=density.deps,
        func=_PartialTraceCallable(density, dimensions, traced),
        metadata={},
    )


def embed_operator(
    source: DomainFunction,
    /,
    *,
    subsystem: int,
    subsystem_dims: Sequence[int],
) -> DomainFunction:
    r"""Embed a local operator into a composite Hilbert space with identities.

    ``subsystem`` uses zero-based indexing into ``subsystem_dims``. The source matrix
    dimension must equal the selected subsystem dimension.
    """
    if not isinstance(source, DomainFunction):
        raise TypeError("embed_operator source must be a DomainFunction.")
    dimensions = _subsystem_dimensions(subsystem_dims)
    selected = _subsystem_index(
        subsystem,
        count=len(dimensions),
        role="subsystem",
    )
    return DomainFunction(
        domain=source.domain,
        deps=source.deps,
        func=_EmbeddedOperatorCallable(source, dimensions, selected),
        metadata={},
    )


__all__ = ["embed_operator", "partial_trace", "tensor_product"]
