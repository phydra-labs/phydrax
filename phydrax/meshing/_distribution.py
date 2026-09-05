#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import operator
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellPartition, PointCloudPlan, PreparedTensorGrid
from ..discretization.fem import (
    FiniteElementDiscretization,
    FiniteElementDistributedPhasePlan,
    FiniteElementPartitionWorksetPlan,
)
from ..discretization.finite_volume import (
    FiniteVolumeDecompositionPlan,
    FiniteVolumeDiscretization,
)
from ..discretization.iga import IsogeometricPlan
from ._assembly import MeshPart
from ._result import CellMeshingResult


def _native_cell_ids(part: MeshPart) -> np.ndarray:
    carrier = part.carrier
    if isinstance(carrier, CellMeshingResult):
        return np.concatenate(
            tuple(np.asarray(block.global_ids) for block in carrier.mesh.blocks)
        )
    if isinstance(carrier, PreparedTensorGrid):
        return np.arange(prod(carrier.cells().shape), dtype=np.int64)
    if isinstance(carrier, PointCloudPlan):
        return np.arange(carrier.points.shape[0], dtype=np.int64)
    return np.arange(carrier.topology.cell_count, dtype=np.int64)


def _tensor_owners(shape: tuple[int, ...], splits: tuple[int, ...]) -> np.ndarray:
    if len(splits) != len(shape) or any(
        split <= 0 or size % split for size, split in zip(shape, splits, strict=True)
    ):
        raise ValueError(
            "Cartesian split factors must divide the native cell shape exactly."
        )
    local = tuple(size // split for size, split in zip(shape, splits, strict=True))
    owners = np.empty(shape, dtype=np.int32)
    for rank, route in enumerate(np.ndindex(splits)):
        slices = tuple(
            slice(index * size, (index + 1) * size)
            for index, size in zip(route, local, strict=True)
        )
        owners[slices] = rank
    return owners.reshape(-1)


def _adjacency(part: MeshPart, count: int) -> list[set[int]]:
    carrier = part.carrier
    neighbors: list[set[int]] = [set() for _ in range(count)]
    if isinstance(carrier, CellMeshingResult):
        # Reuse the native FE topology route, including packed polyhedral faces.
        from ..discretization.fem._generic import _facet_routes

        left, right, _, _ = _facet_routes(carrier.mesh)
        for first, second in zip(left, right, strict=True):
            if second >= 0:
                neighbors[int(first)].add(int(second))
                neighbors[int(second)].add(int(first))
    elif isinstance(carrier, (PreparedTensorGrid, IsogeometricPlan)):
        if isinstance(carrier, PreparedTensorGrid):
            shape = carrier.cells().shape
            periodic = tuple(axis.periodic for axis in carrier.structured_axes)
        else:
            shape = carrier.topology.span_shape
            periodic = (False,) * len(shape)
        for cell, route in enumerate(np.ndindex(shape)):
            for axis, size in enumerate(shape):
                for offset in (-1, 1):
                    adjacent = route[axis] + offset
                    if not periodic[axis] and not 0 <= adjacent < size:
                        continue
                    target = list(route)
                    target[axis] = adjacent % size
                    neighbor = int(np.ravel_multi_index(tuple(target), shape))
                    if neighbor != cell:
                        neighbors[cell].add(neighbor)
    return neighbors


class MeshDistribution(StrictModule, NonTrainableState):
    """Revision-bound native ownership and halo residency, independent of a solver.

    Ownership is normalized into the carrier's native cell/span/point ordering.
    Halo IDs are global, sorted, unique and never locally owned. Compact carriers
    remain intact; only the supplied ownership and communication routes are explicit.
    """

    part: MeshPart
    partition: CellPartition
    cell_global_ids: Array
    owned_rows: tuple[Array, ...]
    halo_rows: tuple[Array, ...]
    halo_global_ids: tuple[Array, ...]
    dependencies: Array
    split_factors: tuple[int, ...] | None = eqx.field(static=True)
    halo_width: int = eqx.field(static=True)
    distribution_id: str = eqx.field(static=True)

    def __init__(
        self,
        part: MeshPart,
        partition: CellPartition,
        /,
        *,
        cell_global_ids: ArrayLike | None = None,
        halo_global_ids: tuple[ArrayLike, ...] | None = None,
        split_factors: tuple[int, ...] | None = None,
        halo_width: int = 1,
    ):
        if not isinstance(part, MeshPart) or not isinstance(partition, CellPartition):
            raise TypeError("Mesh distribution requires MeshPart and CellPartition.")
        width = operator.index(halo_width)
        if isinstance(halo_width, bool) or width < 0:
            raise ValueError("Halo width must be a non-negative integer.")
        native = _native_cell_ids(part)
        ids = native if cell_global_ids is None else np.asarray(cell_global_ids)
        if (
            ids.shape != native.shape
            or not np.issubdtype(ids.dtype, np.integer)
            or not np.array_equal(np.sort(ids), np.sort(native))
        ):
            raise ValueError(
                "Distribution cell IDs must cover the exact native global IDs once."
            )
        if partition.cell_owner.shape != native.shape:
            raise ValueError("Ownership must cover the exact native cell count.")
        lookup = {int(value): row for row, value in enumerate(ids)}
        order = np.asarray([lookup[int(value)] for value in native], dtype=np.int32)
        normalized = CellPartition(
            np.asarray(partition.cell_owner)[order], partition.part_count
        )
        owner = np.asarray(normalized.cell_owner)
        splits = (
            None
            if split_factors is None
            else tuple(operator.index(value) for value in split_factors)
        )
        if splits is not None:
            if not isinstance(part.carrier, PreparedTensorGrid):
                raise TypeError(
                    "Cartesian split factors require a compact tensor carrier."
                )
            expected = _tensor_owners(part.carrier.cells().shape, splits)
            if prod(splits) != partition.part_count or not np.array_equal(
                owner, expected
            ):
                raise ValueError(
                    "Cartesian split factors do not reproduce the supplied ownership."
                )
        if (
            isinstance(part.carrier, PointCloudPlan)
            and partition.part_count > 1
            and halo_global_ids is None
        ):
            raise ValueError(
                "Distributed point carriers require explicit stencil halo residency."
            )
        neighbors = (
            _adjacency(part, native.size)
            if width > 0
            else [set() for _ in range(native.size)]
        )
        required: list[np.ndarray] = []
        owned_rows: list[np.ndarray] = []
        for rank in range(partition.part_count):
            owned = np.flatnonzero(owner == rank)
            owned = owned[np.argsort(native[owned], kind="stable")]
            owned_rows.append(owned.astype(np.int32))
            visited = set(owned.tolist())
            frontier = visited.copy()
            for _ in range(width):
                adjacent = (
                    set().union(*(neighbors[cell] for cell in frontier))
                    if frontier
                    else set()
                )
                frontier = adjacent - visited
                visited.update(frontier)
            remote = np.asarray(
                sorted(visited - set(owned.tolist()), key=lambda row: int(native[row])),
                dtype=np.int32,
            )
            required.append(native[remote])
        halos = (
            tuple(required)
            if halo_global_ids is None
            else tuple(np.asarray(value) for value in halo_global_ids)
        )
        if len(halos) != partition.part_count:
            raise ValueError(
                "Exactly one halo residency vector is required per partition."
            )
        native_rows = {int(value): row for row, value in enumerate(native)}
        halo_rows: list[np.ndarray] = []
        normalized_halos: list[np.ndarray] = []
        dependencies = np.zeros((partition.part_count, partition.part_count), dtype=bool)
        for rank, halo in enumerate(halos):
            if (
                halo.ndim != 1
                or (halo.size and not np.issubdtype(halo.dtype, np.integer))
                or np.unique(halo).size != halo.size
                or not np.all(np.isin(halo, native))
            ):
                raise ValueError("Halo residency requires unique native global IDs.")
            halo = np.sort(halo.astype(np.int64, copy=False), kind="stable")
            rows = np.asarray([native_rows[int(value)] for value in halo], dtype=np.int32)
            if np.any(owner[rows] == rank):
                raise ValueError("Halo residency cannot include locally owned entities.")
            if not np.all(np.isin(required[rank], halo)):
                raise ValueError(
                    "Halo residency omits the required native adjacency reach."
                )
            dependencies[rank, np.unique(owner[rows])] = True
            halo_rows.append(rows)
            normalized_halos.append(halo)
        self.part = part
        self.partition = normalized
        self.cell_global_ids = jnp.asarray(native, dtype=jnp.int64)
        self.owned_rows = tuple(jnp.asarray(rows) for rows in owned_rows)
        self.halo_rows = tuple(jnp.asarray(rows) for rows in halo_rows)
        self.halo_global_ids = tuple(
            jnp.asarray(ids, dtype=jnp.int64) for ids in normalized_halos
        )
        self.dependencies = jnp.asarray(dependencies)
        self.split_factors = splits
        self.halo_width = width
        self.distribution_id = canonical_fingerprint(
            {
                "kind": "mesh-distribution",
                "part": part.part_id,
                "partition": normalized.partition_id,
                "cell_ids": array_tree_fingerprint(native),
                "halos": array_tree_fingerprint(tuple(normalized_halos)),
                "split_factors": splits,
                "halo_width": width,
            }
        )

    @classmethod
    def cartesian(
        cls, part: MeshPart, split_factors: tuple[int, ...], /, *, halo_width: int = 1
    ) -> MeshDistribution:
        if not isinstance(part, MeshPart) or not isinstance(
            part.carrier, PreparedTensorGrid
        ):
            raise TypeError("Cartesian distribution requires a compact tensor MeshPart.")
        splits = tuple(operator.index(value) for value in split_factors)
        owner = _tensor_owners(part.carrier.cells().shape, splits)
        return cls(
            part,
            CellPartition(owner, prod(splits)),
            split_factors=splits,
            halo_width=halo_width,
        )

    def require_current(self, part: MeshPart, /) -> None:
        if (
            not isinstance(part, MeshPart)
            or part.name != self.part.name
            or part.part_id != self.part.part_id
        ):
            raise ValueError("Mesh distribution is stale or belongs to another part.")

    def gather(
        self, rank: int, values: ArrayLike, /, *, include_halo: bool = True
    ) -> Array:
        """Gather owned entities then halo entities, each in increasing global-ID order."""
        rank_ = operator.index(rank)
        field = jnp.asarray(values)
        if not 0 <= rank_ < self.partition.part_count:
            raise ValueError("Distribution rank is out of range.")
        if field.ndim == 0 or field.shape[0] != self.cell_global_ids.size:
            raise ValueError("Distributed fields must follow native cell order.")
        rows = (
            jnp.concatenate((self.owned_rows[rank_], self.halo_rows[rank_]))
            if include_halo
            else self.owned_rows[rank_]
        )
        return field[rows]

    def lower_finite_element(
        self, part: MeshPart, discretization: FiniteElementDiscretization, /
    ) -> FiniteElementDistributedPhasePlan:
        """Lower exact ownership/residency to native FE worksets and interface phases."""
        self.require_current(part)
        if not isinstance(part.carrier, CellMeshingResult) or not isinstance(
            discretization, FiniteElementDiscretization
        ):
            raise TypeError(
                "FE lowering requires a certified cell part and prepared FE discretization."
            )
        result = part.carrier
        if discretization.mesh.mesh_id != result.mesh.mesh_id or array_tree_fingerprint(
            discretization.mesh
        ) != array_tree_fingerprint(result.mesh):
            raise ValueError(
                "FE discretization does not use the exact distribution mesh revision."
            )
        if (
            discretization.default_runtime.geometry_layout_id
            != result.geometry.geometry_layout_id
            or not np.array_equal(
                np.asarray(discretization.default_runtime.coordinates),
                np.asarray(result.geometry.coordinates),
            )
        ):
            raise ValueError(
                "FE discretization geometry differs from the certified part revision."
            )
        shape = (self.partition.part_count, self.cell_global_ids.size)
        owned = np.full(shape, -1, dtype=np.int32)
        halo = np.full(shape, -1, dtype=np.int32)
        for rank, (owned_rows, halo_rows) in enumerate(
            zip(self.owned_rows, self.halo_rows, strict=True)
        ):
            owned[rank, : owned_rows.size] = np.asarray(owned_rows)
            halo[rank, : halo_rows.size] = np.asarray(halo_rows)
        worksets = FiniteElementPartitionWorksetPlan(
            self.partition,
            owned,
            owned >= 0,
            halo,
            halo >= 0,
            self.dependencies,
            self.dependencies.T,
        )
        return FiniteElementDistributedPhasePlan(
            discretization, self.partition, worksets=worksets
        )

    def lower_finite_volume(
        self, part: MeshPart, discretization: FiniteVolumeDiscretization, /
    ) -> FiniteVolumeDecompositionPlan:
        """Lower Cartesian ownership to a real named-sharding FV execution plan."""
        self.require_current(part)
        if not isinstance(part.carrier, PreparedTensorGrid) or not isinstance(
            discretization, FiniteVolumeDiscretization
        ):
            raise TypeError(
                "Structured FV lowering requires a compact tensor part and prepared FV discretization."
            )
        grid = part.carrier
        revision = canonical_fingerprint(array_tree_fingerprint(grid))
        if (
            discretization.grid.prepared_id != grid.prepared_id
            or canonical_fingerprint(array_tree_fingerprint(discretization.grid))
            != revision
        ):
            raise ValueError(
                "FV discretization does not use the exact distribution grid revision."
            )
        splits = self.split_factors
        if splits is None:
            if self.partition.part_count != 1:
                raise ValueError("FV lowering requires explicit Cartesian split factors.")
            splits = (1,) * len(grid.axis_names)
        return FiniteVolumeDecompositionPlan(
            grid.cells().shape,
            splits,
            grid.axis_names,
            halo_width=self.halo_width,
            periodic=tuple(axis.periodic for axis in grid.structured_axes),
            grid_revision=revision,
        )


__all__ = ["MeshDistribution"]
