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
from ._ir import PDEProblemIR
from ._serialize import _canonical_expression, pde_ir_hash
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
    "attribute",
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
PDE_TOKEN_ATTRIBUTES = (
    "none",
    "coordinate_space",
    "coordinate_time",
    "periodic",
    "bounds_absent",
    "lower_bound",
    "upper_bound",
    "representation_scalar",
    "representation_pseudoscalar",
    "representation_vector",
    "representation_pseudovector",
    "representation_tensor",
    "representation_pseudotensor",
    "coordinate_dependency",
    "scale",
    "component_name",
    "parameter_value_absent",
    "parameter_scalar_value",
    "parameter_component_value",
    "functional",
    "region_interior",
    "region_boundary",
    "region_interface",
    "region_initial",
    "region_coordinate",
    "region_component_absent",
    "region_component",
    "condition_initial",
    "condition_boundary",
    "condition_interface",
    "condition_region",
    "condition_coordinate_absent",
    "condition_coordinate",
    "order",
    "axis",
    "nondimensionalization",
)
_KIND_INDEX = {name: index for index, name in enumerate(PDE_TOKEN_KINDS)}
_OPERATOR_INDEX = {
    name: index for index, name in enumerate(PDE_OPERATOR_VOCABULARY)
}
_ATTRIBUTE_INDEX = {
    name: index for index, name in enumerate(PDE_TOKEN_ATTRIBUTES)
}


class PDETokenBatch(StrictModule):
    """Dense, mask-padded tensor encoding of one or more canonical PDE IRs."""

    kind: Array
    operator: Array
    attribute: Array
    symbol: Array
    scalar: Array
    physical_dimension: Array
    slot: Array
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
    names.update(
        component_name
        for item in problem.fields
        for component_name in item.component_names
    )
    names.update(name for name, _ in problem.nondimensionalization)
    return tuple(sorted(names))


def tokenize_pde_ir(
    problem: PDEProblemIR,
    /,
    *,
    max_tokens: int | None = None,
) -> PDETokenBatch:
    """Encode a validated PDE into deterministic canonical tokens and parent links."""
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
    rows: list[
        tuple[
            int,
            int,
            int,
            int,
            float,
            tuple[float, ...],
            int,
            int,
            int,
        ]
    ] = []

    def append(
        kind: str,
        *,
        operator: str = "none",
        attribute: str = "none",
        symbol: str = "",
        scalar: float = 0.0,
        dimension: tuple[float, ...] = (),
        slot: int = -1,
        parent: int = -1,
        depth: int = 0,
    ) -> int:
        index = len(rows)
        rows.append(
            (
                _KIND_INDEX[kind],
                _OPERATOR_INDEX[operator],
                _ATTRIBUTE_INDEX[attribute],
                symbols[symbol],
                float(scalar),
                _dimension(dimension, rank),
                int(slot),
                int(parent),
                int(depth),
            )
        )
        return index

    def append_attribute(
        attribute: str,
        *,
        parent: int,
        depth: int,
        symbol: str = "",
        scalar: float = 0.0,
        slot: int = -1,
    ) -> None:
        append(
            "attribute",
            attribute=attribute,
            symbol=symbol,
            scalar=scalar,
            slot=slot,
            parent=parent,
            depth=depth,
        )

    def append_expression(
        expression: dict[str, Any],
        *,
        parent: int,
        depth: int,
        slot: int,
    ) -> None:
        symbol = (
            expression.get("symbol")
            or expression.get("coordinate")
            or expression.get("region")
            or ""
        )
        index = append(
            "expression",
            operator=expression["op"],
            symbol=symbol,
            scalar=float(expression.get("value", 0.0)),
            dimension=tuple(expression.get("physical_dimension", ())),
            slot=slot,
            parent=parent,
            depth=depth,
        )
        if expression["op"] == "derivative" or "order" in expression:
            append_attribute(
                "order",
                parent=index,
                depth=depth + 1,
                scalar=float(expression.get("order", 1)),
            )
        if "axis" in expression:
            append_attribute(
                "axis",
                parent=index,
                depth=depth + 1,
                scalar=float(expression["axis"]),
            )
        commutative = expression["op"] in ("add", "multiply")
        for argument_slot, argument in enumerate(expression.get("args", ())):
            append_expression(
                argument,
                parent=index,
                depth=depth + 1,
                slot=-1 if commutative else argument_slot,
            )

    for coordinate in sorted(problem.coordinates, key=lambda item: item.name):
        root = append(
            "coordinate",
            symbol=coordinate.name,
            scalar=coordinate.size,
            dimension=coordinate.physical_dimension,
        )
        append_attribute(
            f"coordinate_{coordinate.kind}",
            parent=root,
            depth=1,
        )
        append_attribute(
            "periodic",
            parent=root,
            depth=1,
            scalar=float(coordinate.periodic),
        )
        if coordinate.bounds is None:
            append_attribute("bounds_absent", parent=root, depth=1)
        else:
            append_attribute(
                "lower_bound",
                parent=root,
                depth=1,
                scalar=coordinate.bounds[0],
            )
            append_attribute(
                "upper_bound",
                parent=root,
                depth=1,
                scalar=coordinate.bounds[1],
            )
    for field in sorted(problem.fields, key=lambda item: item.name):
        root = append(
            "field",
            symbol=field.name,
            scalar=field.components,
            dimension=field.physical_dimension,
        )
        append_attribute(
            f"representation_{field.representation}",
            parent=root,
            depth=1,
        )
        for slot, coordinate in enumerate(field.coordinates):
            append_attribute(
                "coordinate_dependency",
                parent=root,
                depth=1,
                symbol=coordinate,
                slot=slot,
            )
        for slot, scale in enumerate(field.scale):
            append_attribute(
                "scale",
                parent=root,
                depth=1,
                scalar=scale,
                slot=slot,
            )
        for slot, component_name in enumerate(field.component_names):
            append_attribute(
                "component_name",
                parent=root,
                depth=1,
                symbol=component_name,
                slot=slot,
            )
    for parameter in sorted(problem.parameters, key=lambda item: item.name):
        root = append(
            "parameter",
            symbol=parameter.name,
            scalar=parameter.components,
            dimension=parameter.physical_dimension,
        )
        if parameter.value is None:
            append_attribute("parameter_value_absent", parent=root, depth=1)
        elif isinstance(parameter.value, (int, float)):
            append_attribute(
                "parameter_scalar_value",
                parent=root,
                depth=1,
                scalar=parameter.value,
            )
        else:
            for slot, value in enumerate(parameter.value):
                append_attribute(
                    "parameter_component_value",
                    parent=root,
                    depth=1,
                    scalar=value,
                    slot=slot,
                )
        for slot, scale in enumerate(parameter.scale):
            append_attribute(
                "scale",
                parent=root,
                depth=1,
                scalar=scale,
                slot=slot,
            )
        append_attribute(
            "functional",
            parent=root,
            depth=1,
            scalar=float(parameter.functional),
        )
    for region in sorted(problem.regions, key=lambda item: item.name):
        root = append(
            "region",
            symbol=region.name,
            scalar=len(region.coordinates),
        )
        append_attribute(f"region_{region.kind}", parent=root, depth=1)
        for slot, coordinate in enumerate(region.coordinates):
            append_attribute(
                "region_coordinate",
                parent=root,
                depth=1,
                symbol=coordinate,
                slot=slot,
            )
        if region.component is None:
            append_attribute("region_component_absent", parent=root, depth=1)
        else:
            append_attribute(
                "region_component",
                parent=root,
                depth=1,
                symbol=region.component,
            )
    for equation in sorted(problem.equations, key=lambda item: item.name):
        root = append("equation", symbol=equation.name)
        append_expression(
            _canonical_expression(equation.lhs),
            parent=root,
            depth=1,
            slot=0,
        )
        append_expression(
            _canonical_expression(equation.rhs),
            parent=root,
            depth=1,
            slot=1,
        )
    for condition in sorted(problem.conditions, key=lambda item: item.name):
        root = append("condition", symbol=condition.name)
        append_attribute(f"condition_{condition.kind}", parent=root, depth=1)
        append_attribute(
            "condition_region",
            parent=root,
            depth=1,
            symbol=condition.region,
        )
        if condition.coordinate is None:
            append_attribute(
                "condition_coordinate_absent",
                parent=root,
                depth=1,
            )
        else:
            append_attribute(
                "condition_coordinate",
                parent=root,
                depth=1,
                symbol=condition.coordinate,
            )
        append_expression(
            _canonical_expression(condition.expression),
            parent=root,
            depth=1,
            slot=0,
        )
        append_expression(
            _canonical_expression(condition.target),
            parent=root,
            depth=1,
            slot=1,
        )
    for name, scale in sorted(problem.nondimensionalization):
        append(
            "attribute",
            attribute="nondimensionalization",
            symbol=name,
            scalar=scale,
        )

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
        [row[5] for row in rows] + [(0.0,) * rank] * padding,
        dtype=float,
    ).reshape((capacity, rank))

    def column(index: int, dtype: Any, padding_value: int | float = 0) -> Array:
        values = [row[index] for row in rows] + [padding_value] * padding
        return jnp.asarray(values, dtype=dtype)

    return PDETokenBatch(
        kind=column(0, jnp.int32),
        operator=column(1, jnp.int32),
        attribute=column(2, jnp.int32),
        symbol=column(3, jnp.int32),
        scalar=column(4, float),
        physical_dimension=dimensions,
        slot=column(6, jnp.int32, -1),
        parent=column(7, jnp.int32, -1),
        depth=column(8, jnp.int32),
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
        attribute=pad(tokens.attribute),
        symbol=pad(tokens.symbol),
        scalar=pad(tokens.scalar),
        physical_dimension=jnp.pad(
            tokens.physical_dimension,
            tuple(dimension_widths),
            constant_values=0.0,
        ),
        slot=pad(tokens.slot, -1),
        parent=pad(tokens.parent, -1),
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
        attribute=jnp.stack(tuple(item.attribute for item in padded)),
        symbol=jnp.stack(tuple(remap_symbols(item) for item in padded)),
        scalar=jnp.stack(tuple(item.scalar for item in padded)),
        physical_dimension=jnp.stack(
            tuple(item.physical_dimension for item in padded)
        ),
        slot=jnp.stack(tuple(item.slot for item in padded)),
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
    "PDE_TOKEN_ATTRIBUTES",
    "PDE_TOKEN_KINDS",
    "PDETokenBatch",
    "pad_pde_tokens",
    "stack_pde_tokens",
    "tokenize_pde_ir",
]
