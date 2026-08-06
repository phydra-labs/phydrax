#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState


def _integer_indices(name: str, value: ArrayLike, /) -> Array:
    indices = jnp.asarray(value)
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise TypeError(f"{name} must have an integer dtype.")
    return indices.astype(jnp.int32)


def _valid_mask(value: ArrayLike | None, shape: tuple[int, ...], /) -> Array:
    if value is None:
        return jnp.ones(shape, dtype=bool)
    valid = jnp.asarray(value, dtype=bool)
    if valid.shape != shape:
        raise ValueError(f"valid must have shape {shape}; got {valid.shape}.")
    return valid


def _check_bounds(
    name: str,
    indices: Array,
    valid: Array,
    size: int,
    /,
) -> Array:
    if int(indices.size) == 0:
        return indices
    return eqx.error_if(
        indices,
        jnp.any(valid & ((indices < 0) | (indices >= int(size)))),
        f"A valid {name} lies outside [0, {int(size)}).",
    )


class EdgeRelation(StrictModule, NonTrainableState):
    """Fixed-capacity source-to-target routes in edge-list form."""

    source_indices: Array
    target_indices: Array
    valid: Array
    source_size: int = eqx.field(static=True)
    target_size: int = eqx.field(static=True)

    def __init__(
        self,
        source_indices: ArrayLike,
        target_indices: ArrayLike,
        /,
        *,
        source_size: int,
        target_size: int,
        valid: ArrayLike | None = None,
    ):
        source_count = int(source_size)
        target_count = int(target_size)
        if source_count < 0 or target_count < 0:
            raise ValueError(
                "Edge relation source and target sizes must be non-negative."
            )

        source = _integer_indices("source_indices", source_indices)
        target = _integer_indices("target_indices", target_indices)
        if source.ndim != 1 or target.ndim != 1:
            raise ValueError("Edge relation indices must be rank-1.")
        if source.shape != target.shape:
            raise ValueError("Edge relation source and target indices must match.")
        if int(source.size) > 0 and (source_count == 0 or target_count == 0):
            raise ValueError(
                "A non-empty edge relation requires non-empty source and target spaces."
            )

        route_valid = _valid_mask(valid, tuple(int(size) for size in source.shape))
        self.source_indices = _check_bounds(
            "edge source index", source, route_valid, source_count
        )
        self.target_indices = _check_bounds(
            "edge target index", target, route_valid, target_count
        )
        self.valid = route_valid
        self.source_size = source_count
        self.target_size = target_count

    @property
    def route_shape(self) -> tuple[int, ...]:
        return (int(self.source_indices.shape[0]),)

    @property
    def capacity(self) -> int:
        return int(self.source_indices.shape[0])

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (self.source_size,)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.target_size,)

    def transpose(self) -> "EdgeRelation":
        """Swap source and target spaces without reordering routes."""
        return EdgeRelation(
            self.target_indices,
            self.source_indices,
            source_size=self.target_size,
            target_size=self.source_size,
            valid=self.valid,
        )

    def with_valid(self, valid: ArrayLike, /) -> "EdgeRelation":
        """Return this relation with an additional route-validity condition."""
        extra = _valid_mask(valid, self.route_shape)
        return EdgeRelation(
            self.source_indices,
            self.target_indices,
            source_size=self.source_size,
            target_size=self.target_size,
            valid=self.valid & extra,
        )


class RowRelation(StrictModule, NonTrainableState):
    """Fixed-width, case-local source routes grouped by target rows."""

    source_indices: Array
    valid: Array
    source_size: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        source_indices: ArrayLike,
        /,
        *,
        source_size: int,
        valid: ArrayLike | None = None,
        case_shape: tuple[int, ...] = (),
    ):
        source_count = int(source_size)
        if source_count <= 0:
            raise ValueError("Row relation source_size must be positive.")
        cases = tuple(int(size) for size in case_shape)
        if any(size <= 0 for size in cases):
            raise ValueError("Row relation case dimensions must be positive.")

        source = _integer_indices("source_indices", source_indices)
        if source.ndim <= len(cases):
            raise ValueError(
                "Row relation indices must contain target rows and a route-width axis."
            )
        if tuple(int(size) for size in source.shape[: len(cases)]) != cases:
            raise ValueError(
                f"Row relation indices must begin with case_shape {cases}; "
                f"got {source.shape}."
            )
        if int(source.shape[-1]) <= 0:
            raise ValueError("Row relation route width must be positive.")

        route_valid = _valid_mask(valid, tuple(int(size) for size in source.shape))
        self.source_indices = _check_bounds(
            "row source index", source, route_valid, source_count
        )
        self.valid = route_valid
        self.source_size = source_count
        self.case_shape = cases

    @property
    def route_shape(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.source_indices.shape)

    @property
    def target_shape(self) -> tuple[int, ...]:
        return self.route_shape[len(self.case_shape) : -1]

    @property
    def width(self) -> int:
        return int(self.source_indices.shape[-1])

    @property
    def num_cases(self) -> int:
        return prod(self.case_shape) if self.case_shape else 1

    @property
    def targets_per_case(self) -> int:
        return prod(self.target_shape) if self.target_shape else 1

    @property
    def capacity(self) -> int:
        return int(self.source_indices.size)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.case_shape + (self.source_size,)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.case_shape + self.target_shape

    def with_valid(self, valid: ArrayLike, /) -> "RowRelation":
        """Return this relation with an additional route-validity condition."""
        extra = _valid_mask(valid, self.route_shape)
        return RowRelation(
            self.source_indices,
            source_size=self.source_size,
            valid=self.valid & extra,
            case_shape=self.case_shape,
        )

    def as_edge_relation(self) -> EdgeRelation:
        """Flatten cases and target rows into one disconnected edge relation."""
        cases = self.num_cases
        targets = self.targets_per_case
        width = self.width
        local_source = self.source_indices.reshape((cases, targets, width))
        source_offsets = (jnp.arange(cases, dtype=jnp.int32) * self.source_size).reshape(
            (cases, 1, 1)
        )
        source = local_source + source_offsets
        target = jnp.broadcast_to(
            (
                jnp.arange(cases, dtype=jnp.int32)[:, None] * targets
                + jnp.arange(targets, dtype=jnp.int32)[None, :]
            )[..., None],
            (cases, targets, width),
        )
        return EdgeRelation(
            source.reshape((-1,)),
            target.reshape((-1,)),
            source_size=cases * self.source_size,
            target_size=cases * targets,
            valid=self.valid.reshape((-1,)),
        )


SparseRelation: TypeAlias = EdgeRelation | RowRelation


__all__ = ["EdgeRelation", "RowRelation", "SparseRelation"]
