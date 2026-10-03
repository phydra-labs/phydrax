#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState


# Concrete (host-prepared) topology is converted and validated with NumPy once
# at construction and placed with ``device_put`` (a transfer, not a compiled
# program); traced topology keeps device-side conversion and checks inside the
# caller's compilation.


def _traced(*values: object) -> bool:
    return any(isinstance(value, jax_core.Tracer) for value in values)


def _integer_indices(name: str, value: ArrayLike, /) -> Array:
    if isinstance(value, jax_core.Tracer):
        if not jnp.issubdtype(value.dtype, jnp.integer):
            raise TypeError(f"{name} must have an integer dtype.")
        return value.astype(jnp.int32)
    if isinstance(value, jax.Array) and value.dtype == jnp.int32:
        return value
    host = np.asarray(value)
    if not np.issubdtype(host.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype.")
    return jax.device_put(host.astype(np.int32))


def _valid_mask(value: ArrayLike | None, shape: tuple[int, ...], /) -> Array:
    if value is None:
        return jax.device_put(np.ones(shape, dtype=np.bool_))
    if isinstance(value, jax_core.Tracer):
        valid = value.astype(jnp.bool_)
    elif isinstance(value, jax.Array) and value.dtype == jnp.bool_:
        valid = value
    else:
        valid = jax.device_put(np.asarray(value, dtype=np.bool_))
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
    if indices.size == 0:
        return indices
    message = f"A valid {name} lies outside [0, {size})."
    if _traced(indices, valid):
        invalid = jnp.any(valid & ((indices < 0) | (indices >= size)))
        return eqx.error_if(indices, invalid, message)
    # Immutable symbolic admission is a host decision at construction.
    host_indices, host_valid = jax.device_get((indices, valid))
    if np.any(host_valid & ((host_indices < 0) | (host_indices >= size))):
        raise ValueError(message)
    return indices


def _and_valid(first: Array, second: Array, /) -> Array:
    if _traced(first, second):
        return first & second
    host_first, host_second = jax.device_get((first, second))
    return jax.device_put(np.logical_and(host_first, host_second))


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
    ) -> None:
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
        if source.size > 0 and (source_count == 0 or target_count == 0):
            raise ValueError(
                "A non-empty edge relation requires non-empty source and target spaces."
            )

        route_valid = _valid_mask(valid, tuple(source.shape))
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
        return (self.source_indices.shape[0],)

    @property
    def capacity(self) -> int:
        return self.source_indices.shape[0]

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
            valid=_and_valid(self.valid, extra),
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
    ) -> None:
        source_count = int(source_size)
        if source_count <= 0:
            raise ValueError("Row relation source_size must be positive.")
        cases = tuple(case_shape)
        if any(size <= 0 for size in cases):
            raise ValueError("Row relation case dimensions must be positive.")

        source = _integer_indices("source_indices", source_indices)
        if source.ndim <= len(cases):
            raise ValueError(
                "Row relation indices must contain target rows and a route-width axis."
            )
        if tuple(source.shape[: len(cases)]) != cases:
            raise ValueError(
                f"Row relation indices must begin with case_shape {cases}; got {source.shape}."
            )
        if source.shape[-1] <= 0:
            raise ValueError("Row relation route width must be positive.")

        route_valid = _valid_mask(valid, tuple(source.shape))
        self.source_indices = _check_bounds(
            "row source index", source, route_valid, source_count
        )
        self.valid = route_valid
        self.source_size = source_count
        self.case_shape = cases

    @property
    def route_shape(self) -> tuple[int, ...]:
        return tuple(self.source_indices.shape)

    @property
    def target_shape(self) -> tuple[int, ...]:
        return self.route_shape[len(self.case_shape) : -1]

    @property
    def width(self) -> int:
        return self.source_indices.shape[-1]

    @property
    def num_cases(self) -> int:
        return prod(self.case_shape) if self.case_shape else 1

    @property
    def targets_per_case(self) -> int:
        return prod(self.target_shape) if self.target_shape else 1

    @property
    def capacity(self) -> int:
        return self.source_indices.size

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
            valid=_and_valid(self.valid, extra),
            case_shape=self.case_shape,
        )

    def as_edge_relation(self) -> EdgeRelation:
        """Flatten cases and target rows into one disconnected edge relation."""
        cases = self.num_cases
        targets = self.targets_per_case
        width = self.width
        if _traced(self.source_indices, self.valid):
            xp, indices, valid = jnp, self.source_indices, self.valid
        else:
            # Concrete topology is flattened on the host: no shape-specific
            # device programs are compiled for a preparation-time view.
            xp = np
            indices, valid = jax.device_get((self.source_indices, self.valid))
        local_source = indices.reshape((cases, targets, width))
        source_offsets = (xp.arange(cases, dtype=xp.int32) * self.source_size).reshape(
            (cases, 1, 1)
        )
        source = local_source + source_offsets
        target = xp.broadcast_to(
            (
                xp.arange(cases, dtype=xp.int32)[:, None] * targets
                + xp.arange(targets, dtype=xp.int32)[None, :]
            )[..., None],
            (cases, targets, width),
        )
        return EdgeRelation(
            source.reshape((-1,)),
            target.reshape((-1,)),
            source_size=cases * self.source_size,
            target_size=cases * targets,
            valid=valid.reshape((-1,)),
        )


SparseRelation: TypeAlias = EdgeRelation | RowRelation


def _relation_traced(relation: SparseRelation, /) -> bool:
    """Whether any topology array of ``relation`` is a tracer."""
    if isinstance(relation, RowRelation):
        return _traced(relation.source_indices, relation.valid)
    return _traced(relation.source_indices, relation.target_indices, relation.valid)


def _coalesced_route_count(relation: SparseRelation, /) -> int | None:
    """Distinct valid (target, source) pairs of host topology; ``None`` when traced.

    This is the canonical sparse storage entry count, read once from concrete
    host topology so traced executions never need it from device data.
    """
    if _relation_traced(relation):
        return None
    if isinstance(relation, RowRelation):
        indices, valid = jax.device_get((relation.source_indices, relation.valid))
        # Each flattened (case, row) is one distinct target, so coalescing is
        # per-row deduplication of valid sources; invalid slots sort first.
        rows = np.sort(
            np.where(valid, indices, -1).reshape((-1, relation.width)), axis=-1
        )
        return int(
            np.count_nonzero(rows[:, 0] >= 0)
            + np.count_nonzero((rows[:, 1:] != rows[:, :-1]) & (rows[:, 1:] >= 0))
        )
    source, target, valid = jax.device_get(
        (relation.source_indices, relation.target_indices, relation.valid)
    )
    keys = target[valid].astype(np.int64) * relation.source_size + source[valid]
    if keys.size == 0:
        return 0
    if np.any(keys[1:] < keys[:-1]):
        keys = np.sort(keys)
    return int(1 + np.count_nonzero(keys[1:] != keys[:-1]))


__all__ = ["EdgeRelation", "RowRelation", "SparseRelation"]
