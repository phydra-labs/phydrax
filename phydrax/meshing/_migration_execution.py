# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Ephemeral execution of historical logical ranks on a real physical mesh.

The saved rank axis is numerical data, not a device axis. Named batching keeps
axis_index, ordered ppermute packets and consensus reductions identical to the
physical migration kernels; the compiler distributes those rows over the actual
mesh. Neither this backend nor its device handles belongs in an archive.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, TYPE_CHECKING

import equinox as eqx
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import phydrax

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._adaptive_simplex import AdaptiveSimplexParts

if TYPE_CHECKING:
    from ._migration_record import MigrationPlacement


class LogicalMigrationParts(StrictModule, NonTrainableState):
    """Historical owner identity, separate from its current physical placement."""

    placement: phydrax.meshing._migration_record.MigrationPlacement = eqx.field(static=True)
    mesh: Mesh = eqx.field(static=True)
    logical_rank_ids: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, placement: MigrationPlacement, mesh: Mesh, /) -> None:
        placement.validate_restored()
        if (
            mesh.axis_names != (placement.axis_name,)
            or mesh.devices.size < 1
            or placement.part_count % mesh.devices.size
            or not set(mesh.devices.flat) <= set(jax.devices())
        ):
            raise ValueError(
                "LogicalMigrationParts.mesh must place whole historical rank rows on actual current devices; got incompatible physical placement"
            )
        self.placement = placement
        self.mesh = mesh
        self.logical_rank_ids = tuple(range(placement.part_count))

    @property
    def part_count(self) -> int:
        return len(self.logical_rank_ids)

    @property
    def axis_name(self) -> str:
        return self.placement.axis_name

    @property
    def neighbor_pairs(self) -> tuple[tuple[int, int], ...]:
        return self.placement.neighbor_pairs


MigrationParts: TypeAlias = AdaptiveSimplexParts | LogicalMigrationParts


def partition_map(
    function: Callable,
    *,
    parts: MigrationParts,
    in_specs,
    out_specs,
    check_vma: bool = True,
) -> Callable:
    """Run the same singleton-block kernel on physical or historical rank rows.

    Only the migration substrate's leading owner axis and replicated metadata
    are admitted. Replicated outputs must actually be unbatched (e.g. psum), as
    enforced by vmap's out_axes=None rather than taking an arbitrary rank copy.
    """
    if isinstance(parts, AdaptiveSimplexParts):
        return jax.shard_map(
            function,
            mesh=parts.mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

    def axis(spec: PartitionSpec) -> int | None:
        if spec == PartitionSpec(parts.axis_name):
            return 0
        if spec == PartitionSpec():
            return None
        raise ValueError(
            f"migration.partition_map.spec must describe the historical leading rank axis or replication; got {spec}"
        )

    is_spec = lambda value: isinstance(value, PartitionSpec)
    in_axes = jax.tree.map(axis, in_specs, is_leaf=is_spec)
    out_axes = jax.tree.map(axis, out_specs, is_leaf=is_spec)
    is_axis = lambda value: value is None or type(value) is int

    def local(*rows):
        blocks = jax.tree.map(
            lambda index, value: jax.tree.map(lambda leaf: leaf[None], value)
            if index == 0
            else value,
            in_axes,
            rows,
            is_leaf=is_axis,
        )
        result = function(*blocks)
        return jax.tree.map(
            lambda index, value: jax.tree.map(lambda leaf: leaf[0], value)
            if index == 0
            else value,
            out_axes,
            result,
            is_leaf=is_axis,
        )

    mapped = jax.vmap(
        local,
        in_axes=in_axes,
        out_axes=out_axes,
        axis_name=parts.axis_name,
        axis_size=parts.part_count,
    )

    def execute(*arguments):
        def check(index, value):
            if index == 0 and any(
                not leaf.ndim or leaf.shape[0] != parts.part_count
                for leaf in jax.tree.leaves(value)
            ):
                raise ValueError(
                    "migration.partition_map.rows must preserve every historical logical rank in order; got incompatible leading extent"
                )
        jax.tree.map(check, in_axes, arguments, is_leaf=is_axis)
        result = mapped(*arguments)
        return jax.tree.map(
            lambda index, value: jax.tree.map(
                lambda leaf: jax.lax.with_sharding_constraint(
                    leaf,
                    NamedSharding(
                        parts.mesh,
                        PartitionSpec(parts.axis_name) if index == 0 else PartitionSpec(),
                    ),
                ),
                value,
            ),
            out_axes,
            result,
            is_leaf=is_axis,
        )

    return execute


def with_migration_routes(
    parts: MigrationParts, routes: tuple[tuple[int, int], ...]
) -> MigrationParts:
    """Change measured logical routes without changing historical owner identity."""
    if isinstance(parts, LogicalMigrationParts):
        saved = parts.placement
        placement = type(saved)(
            saved.axis_name,
            saved.axis_names,
            saved.mesh_shape,
            saved.device_keys,
            saved.runtime_device_keys,
            routes,
        )
        return LogicalMigrationParts(placement, parts.mesh)
    return AdaptiveSimplexParts(
        tuple(parts.mesh.devices.flat), axis_name=parts.axis_name, neighbor_pairs=routes
    )
