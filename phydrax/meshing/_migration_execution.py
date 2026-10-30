#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Execute historical logical owner ranks on the currently admitted devices."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast, final, TypeVar

import equinox as eqx
import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from .._execution_runtime import ExecutionGroup
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._adaptive_simplex import AdaptiveSimplexParts


T = TypeVar("T")


def _routes(
    part_count: int, routes: Sequence[tuple[int, int]], /
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    for route in routes:
        if (
            not isinstance(route, tuple)
            or len(route) != 2
            or any(
                isinstance(rank, bool) or not isinstance(rank, (int, np.integer))
                for rank in route
            )
        ):
            raise TypeError("neighbor_pairs must contain integer rank pairs.")
        source, destination = (int(route[0]), int(route[1]))
        if not 0 <= source < part_count or not 0 <= destination < part_count:
            raise ValueError(
                "neighbor_pairs contain a rank outside the logical owner set."
            )
        if source == destination:
            raise ValueError("neighbor_pairs cannot contain self routes.")
        normalized.append((source, destination))
    if len(set(normalized)) != len(normalized):
        raise ValueError("neighbor_pairs must be unique.")
    return tuple(sorted(normalized))


@final
class LogicalMigrationParts(StrictModule, NonTrainableState):
    """Saved logical owner identity bound to a distinct current physical group."""

    execution_group: ExecutionGroup = eqx.field(static=True)
    logical_rank_ids: tuple[int, ...] = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    neighbor_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)

    def __init__(
        self,
        execution_group: ExecutionGroup,
        logical_part_count: int,
        /,
        *,
        axis_name: str = "logical_owner",
        neighbor_pairs: Sequence[tuple[int, int]] = (),
    ) -> None:
        if not isinstance(execution_group, ExecutionGroup):
            raise TypeError("execution_group must be ExecutionGroup.")
        if (
            isinstance(logical_part_count, bool)
            or not isinstance(logical_part_count, (int, np.integer))
            or int(logical_part_count) < 1
        ):
            raise ValueError("logical_part_count must be a positive integer.")
        if not isinstance(axis_name, str) or not axis_name.strip():
            raise ValueError("axis_name must be a non-empty string.")
        if not execution_group.devices or not execution_group.is_member:
            raise ValueError("execution_group must be the current member device group.")
        count = int(logical_part_count)
        self.execution_group = execution_group
        self.logical_rank_ids = tuple(range(count))
        self.axis_name = axis_name.strip()
        self.neighbor_pairs = _routes(count, neighbor_pairs)

    @property
    def part_count(self) -> int:
        """Number of saved logical owner ranks, independent of current devices."""
        return len(self.logical_rank_ids)


type MigrationParts = AdaptiveSimplexParts | LogicalMigrationParts


def _axis_index(specification: PartitionSpec, axis_name: str, /) -> int | None:
    if specification == PartitionSpec(axis_name):
        return 0
    if specification == PartitionSpec():
        return None
    raise ValueError(
        "logical migration supports only the historical leading rank axis or replicated values."
    )


def _replicate_on_current_group(tree: T, execution_group: ExecutionGroup, /) -> T:
    sharding = NamedSharding(execution_group.mesh, PartitionSpec())

    def place(value: object) -> object:
        if isinstance(value, (jax.Array, np.ndarray)):
            return jax.device_put(value, sharding)
        return value

    return cast(T, jax.tree.map(place, tree))


def partition_map(
    function: Callable[..., T],
    *,
    parts: MigrationParts,
    in_specs: object,
    out_specs: object,
    check_vma: bool = True,
) -> Callable[..., T]:
    """Run a part-local kernel over physical shards or saved logical rank rows.

    Logical rank arrays are replicated on the current physical mesh and the
    historical leading axis is executed with ``vmap``. This keeps the saved
    rank count independent of the number of current devices (including ragged
    counts such as three saved ranks on two devices) without serializing device
    handles or requiring an archive-time mesh to exist.
    """
    if isinstance(parts, AdaptiveSimplexParts):
        return jax.shard_map(
            function,
            mesh=parts.mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )
    if not isinstance(parts, LogicalMigrationParts):
        raise TypeError("parts must be AdaptiveSimplexParts or LogicalMigrationParts.")
    spec_leaf = lambda value: isinstance(value, PartitionSpec)
    in_axes = jax.tree.map(
        lambda spec: _axis_index(spec, parts.axis_name),
        in_specs,
        is_leaf=spec_leaf,
    )
    out_axes = jax.tree.map(
        lambda spec: _axis_index(spec, parts.axis_name),
        out_specs,
        is_leaf=spec_leaf,
    )
    axis_leaf = lambda value: value is None or type(value) is int

    def local(*rows: object) -> T:
        blocks = jax.tree.map(
            lambda axis, value: (
                jax.tree.map(lambda leaf: leaf[None], value) if axis == 0 else value
            ),
            in_axes,
            rows,
            is_leaf=axis_leaf,
        )
        result = function(*blocks)
        return jax.tree.map(
            lambda axis, value: (
                jax.tree.map(lambda leaf: leaf[0], value) if axis == 0 else value
            ),
            out_axes,
            result,
            is_leaf=axis_leaf,
        )

    mapped = jax.vmap(
        local,
        in_axes=in_axes,
        out_axes=out_axes,
        axis_name=parts.axis_name,
        axis_size=parts.part_count,
    )

    def execute(*arguments: object) -> T:
        def validate_rows(index: int | None, value: object) -> None:
            if index == 0 and any(
                not leaf.ndim or leaf.shape[0] != parts.part_count
                for leaf in jax.tree.leaves(value)
                if isinstance(leaf, (jax.Array, np.ndarray))
            ):
                raise ValueError(
                    "logical migration arrays must retain every historical rank in order."
                )

        jax.tree.map(validate_rows, in_axes, arguments, is_leaf=axis_leaf)
        placed = _replicate_on_current_group(arguments, parts.execution_group)
        result = mapped(*placed)
        return _replicate_on_current_group(result, parts.execution_group)

    return execute


def with_migration_routes(
    parts: MigrationParts, routes: Sequence[tuple[int, int]], /
) -> MigrationParts:
    """Bind a deterministic routing order without changing logical rank identity."""
    if isinstance(parts, LogicalMigrationParts):
        return LogicalMigrationParts(
            parts.execution_group,
            parts.part_count,
            axis_name=parts.axis_name,
            neighbor_pairs=routes,
        )
    if isinstance(parts, AdaptiveSimplexParts):
        raise ValueError(
            "Physical migration routes are owned by the adaptive execution mesh."
        )
    raise TypeError("parts must be AdaptiveSimplexParts or LogicalMigrationParts.")


__all__ = [
    "LogicalMigrationParts",
    "MigrationParts",
    "partition_map",
    "with_migration_routes",
]
