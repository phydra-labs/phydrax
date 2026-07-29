#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import replace
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from ._ir import PDEExpression, PDEProblemIR
from ._serialize import pde_ir_hash
from ._validate import validate_pde_ir


PDE_TOKEN_KINDS = (
    "padding",
    "coordinate",
    "field",
    "parameter",
    "region",
    "equation",
    "condition",
    "expression",
)
PDE_OPERATOR_VOCABULARY = (
    "none",
    "constant",
    "coordinate",
    "field",
    "parameter",
    "add",
    "multiply",
    "divide",
    "negate",
    "power",
    "sin",
    "cos",
    "exp",
    "log",
    "sqrt",
    "component",
    "dot",
    "derivative",
    "gradient",
    "divergence",
    "curl",
    "laplacian",
    "integral",
)
_KIND_INDEX = {name: index for index, name in enumerate(PDE_TOKEN_KINDS)}
_OPERATOR_INDEX = {
    name: index for index, name in enumerate(PDE_OPERATOR_VOCABULARY)
}


class PDETokenBatch(StrictModule):
    """Dense, mask-padded tensor encoding of one or more canonical PDE IRs."""

    kind: Array
    operator: Array
    symbol: Array
    scalar: Array
    physical_dimension: Array
    parent: Array
    depth: Array
    mask: Array
    symbol_vocabulary: tuple[str, ...] = eqx.field(static=True)
    canonical_hashes: tuple[str, ...] = eqx.field(static=True)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.mask.shape[:-1])

    @property
    def max_tokens(self) -> int:
        return int(self.mask.shape[-1])


def _dimension(values: tuple[float, ...], rank: int, /) -> tuple[float, ...]:
    return (0.0,) * rank if not values else tuple(values)


def _symbol_vocabulary(problem: PDEProblemIR, /) -> tuple[str, ...]:
    names = {""}
    names.update(item.name for item in problem.coordinates)
    names.update(item.name for item in problem.fields)
    names.update(item.name for item in problem.parameters)
    names.update(item.name for item in problem.regions)
    names.update(item.name for item in problem.equations)
    names.update(item.name for item in problem.conditions)
    names.update(
        item.component
        for item in problem.regions
        if item.component is not None
    )
    return tuple(sorted(names))


def tokenize_pde_ir(
    problem: PDEProblemIR,
    /,
    *,
    max_tokens: int | None = None,
) -> PDETokenBatch:
    """Encode a validated PDE into deterministic preorder tokens and parent links."""
    validate_pde_ir(problem)
    rank = max(
        (
            len(item.physical_dimension)
            for item in (
                *problem.coordinates,
                *problem.fields,
                *problem.parameters,
            )
        ),
        default=0,
    )
    vocabulary = _symbol_vocabulary(problem)
    symbols = {name: index for index, name in enumerate(vocabulary)}
    rows: list[tuple[int, int, int, float, tuple[float, ...], int, int]] = []

    def append(
        kind: str,
        *,
        operator: str = "none",
        symbol: str = "",
        scalar: float = 0.0,
        dimension: tuple[float, ...] = (),
        parent: int = -1,
        depth: int = 0,
    ) -> int:
        index = len(rows)
        rows.append(
            (
                _KIND_INDEX[kind],
                _OPERATOR_INDEX[operator],
                symbols[symbol],
                float(scalar),
                _dimension(dimension, rank),
                int(parent),
                int(depth),
            )
        )
        return index

    def append_expression(
        expression: PDEExpression,
        *,
        parent: int,
        depth: int,
    ) -> None:
        symbol = expression.symbol or expression.coordinate or expression.region or ""
        index = append(
            "expression",
            operator=expression.op,
            symbol=symbol,
            scalar=0.0 if expression.value is None else expression.value,
            dimension=expression.physical_dimension,
            parent=parent,
            depth=depth,
        )
        for argument in expression.args:
            append_expression(argument, parent=index, depth=depth + 1)

    for coordinate in sorted(problem.coordinates, key=lambda item: item.name):
        append(
            "coordinate",
            symbol=coordinate.name,
            scalar=coordinate.size,
            dimension=coordinate.physical_dimension,
        )
    for field in sorted(problem.fields, key=lambda item: item.name):
        append(
            "field",
            symbol=field.name,
            scalar=field.components,
            dimension=field.physical_dimension,
        )
    for parameter in sorted(problem.parameters, key=lambda item: item.name):
        scalar = parameter.value if isinstance(parameter.value, float) else 0.0
        append(
            "parameter",
            symbol=parameter.name,
            scalar=scalar,
            dimension=parameter.physical_dimension,
        )
    for region in sorted(problem.regions, key=lambda item: item.name):
        append(
            "region",
            symbol=region.name,
            scalar=len(region.coordinates),
        )
    for equation in sorted(problem.equations, key=lambda item: item.name):
        root = append("equation", symbol=equation.name)
        append_expression(equation.lhs, parent=root, depth=1)
        append_expression(equation.rhs, parent=root, depth=1)
    for condition in sorted(problem.conditions, key=lambda item: item.name):
        root = append("condition", symbol=condition.name)
        append_expression(condition.expression, parent=root, depth=1)
        append_expression(condition.target, parent=root, depth=1)

    count = len(rows)
    capacity = count if max_tokens is None else int(max_tokens)
    if capacity < count:
        raise ValueError(
            f"PDE token count {count} exceeds requested max_tokens {capacity}."
        )
    if capacity <= 0:
        raise ValueError("PDE token capacity must be positive.")
    padding = capacity - count
    dimensions = jnp.asarray(
        [row[4] for row in rows] + [(0.0,) * rank] * padding,
        dtype=float,
    ).reshape((capacity, rank))

    def column(index: int, dtype: Any) -> Array:
        values = [row[index] for row in rows] + [0] * padding
        return jnp.asarray(values, dtype=dtype)

    return PDETokenBatch(
        kind=column(0, jnp.int32),
        operator=column(1, jnp.int32),
        symbol=column(2, jnp.int32),
        scalar=column(3, float),
        physical_dimension=dimensions,
        parent=column(5, jnp.int32),
        depth=column(6, jnp.int32),
        mask=jnp.arange(capacity) < count,
        symbol_vocabulary=vocabulary,
        canonical_hashes=(pde_ir_hash(problem),),
    )


def pad_pde_tokens(tokens: PDETokenBatch, max_tokens: int, /) -> PDETokenBatch:
    """Right-pad a token batch without changing valid token indices or parents."""
    capacity = int(max_tokens)
    if capacity < tokens.max_tokens:
        raise ValueError("Cannot pad PDE tokens to a smaller capacity.")
    amount = capacity - tokens.max_tokens
    if amount == 0:
        return tokens

    def pad(array: Array, value: int | float | bool = 0) -> Array:
        widths = [(0, 0)] * array.ndim
        widths[-1] = (0, amount)
        return jnp.pad(array, tuple(widths), constant_values=value)

    dimension_widths = [(0, 0)] * tokens.physical_dimension.ndim
    dimension_widths[-2] = (0, amount)
    return replace(
        tokens,
        kind=pad(tokens.kind),
        operator=pad(tokens.operator),
        symbol=pad(tokens.symbol),
        scalar=pad(tokens.scalar),
        physical_dimension=jnp.pad(
            tokens.physical_dimension, tuple(dimension_widths), constant_values=0.0
        ),
        parent=pad(tokens.parent),
        depth=pad(tokens.depth),
        mask=pad(tokens.mask, False),
    )


def stack_pde_tokens(tokens: tuple[PDETokenBatch, ...], /) -> PDETokenBatch:
    """Stack tokenized problems after merging their deterministic vocabularies."""
    if not tokens:
        raise ValueError("stack_pde_tokens requires at least one token batch.")
    capacity = max(item.max_tokens for item in tokens)
    padded = tuple(pad_pde_tokens(item, capacity) for item in tokens)
    vocabulary = tuple(
        sorted({symbol for item in padded for symbol in item.symbol_vocabulary})
    )
    merged_symbols = {name: index for index, name in enumerate(vocabulary)}
    dimension_size = padded[0].physical_dimension.shape[-1]
    if any(item.physical_dimension.shape[-1] != dimension_size for item in padded[1:]):
        raise ValueError("Stacked PDE token batches must share a dimension rank.")

    def remap_symbols(item: PDETokenBatch) -> Array:
        remapping = jnp.asarray(
            [merged_symbols[name] for name in item.symbol_vocabulary],
            dtype=jnp.int32,
        )
        return remapping[item.symbol]

    return PDETokenBatch(
        kind=jnp.stack(tuple(item.kind for item in padded)),
        operator=jnp.stack(tuple(item.operator for item in padded)),
        symbol=jnp.stack(tuple(remap_symbols(item) for item in padded)),
        scalar=jnp.stack(tuple(item.scalar for item in padded)),
        physical_dimension=jnp.stack(
            tuple(item.physical_dimension for item in padded)
        ),
        parent=jnp.stack(tuple(item.parent for item in padded)),
        depth=jnp.stack(tuple(item.depth for item in padded)),
        mask=jnp.stack(tuple(item.mask for item in padded)),
        symbol_vocabulary=vocabulary,
        canonical_hashes=tuple(
            hash_value for item in padded for hash_value in item.canonical_hashes
        ),
    )


__all__ = [
    "PDE_OPERATOR_VOCABULARY",
    "PDE_TOKEN_KINDS",
    "PDETokenBatch",
    "pad_pde_tokens",
    "stack_pde_tokens",
    "tokenize_pde_ir",
]
