#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Mesh ownership, ghost residency, and ownership migration across revisions.

Ownership comes from weighted contiguous ranges along Morton or Hilbert
space-filling curves, from multilevel graph partitioning through the METIS
plugin, or from a provider. Ghost layers are breadth-first face-adjacency
reach. Transitions carry ownership through mesh lineage onto adapted parts,
repartition only when inherited ownership is out of balance, and migrate cell
data through explicit send/receive sparse relations.
"""

from __future__ import annotations

import operator
from enum import StrEnum
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellPartition, PointCloudPlan, PreparedTensorGrid
from ..discretization._partition import (
    CellAdjacency,
    CellPartitionHalo,
    inherit_cell_owners,
    mesh_cell_adjacency,
    padded_part_table,
)
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
from ..discretization.spatial import hilbert_encode_integer, morton_encode_integer
from ..sparse import EdgeRelation, gather_routes, route_reduce, RouteReduction
from ._assembly import MeshPart
from ._lineage import MeshLineage
from ._metis import metis_identity, metis_partition, MetisPartitionError
from ._result import CellMeshingResult


_MAXIMUM_CURVE_DEPTH = 21


class MeshPartitionKind(StrEnum):
    """Ownership route: space-filling curve, graph partitioner, or provider."""

    MORTON = "morton"
    HILBERT = "hilbert"
    GRAPH = "graph"
    PROVIDER = "provider"


class MeshPartitionPolicy(StrictModule, NonTrainableState):
    """How cell ownership is produced, balanced, and ghosted.

    ``maximum_imbalance`` bounds max-part-weight / mean-part-weight: it is the
    METIS balance target and the tolerance under which a transition keeps
    inherited ownership instead of repartitioning. Curve routes guarantee
    ``imbalance <= 1 + part_count * max_cell_weight / total_weight`` whenever
    every part can be non-empty. ``curve_depth`` is the per-axis bit depth of
    the isotropic curve quantization; equal codes order by global ID.
    """

    kind: MeshPartitionKind = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    halo_width: int = eqx.field(static=True)
    maximum_imbalance: float = eqx.field(static=True)
    curve_depth: int = eqx.field(static=True)
    graph_seed: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: MeshPartitionKind,
        part_count: int,
        /,
        *,
        halo_width: int = 1,
        maximum_imbalance: float = 1.05,
        curve_depth: int = 16,
        graph_seed: int = 0,
    ):
        if not isinstance(kind, MeshPartitionKind):
            raise TypeError("kind must be MeshPartitionKind.")
        counts = (part_count, halo_width, curve_depth, graph_seed)
        if any(isinstance(value, bool) for value in counts):
            raise TypeError("Partition policy counts must be integers.")
        parts, width, depth, seed = (operator.index(value) for value in counts)
        tolerance = float(maximum_imbalance)
        if parts <= 0 or width < 0 or seed < 0:
            raise ValueError(
                "Partition count must be positive; halo width and seed non-negative."
            )
        if not 1 <= depth <= _MAXIMUM_CURVE_DEPTH:
            raise ValueError(
                f"Curve depth must lie in [1, {_MAXIMUM_CURVE_DEPTH}] bits per axis."
            )
        if not np.isfinite(tolerance) or tolerance < 1.0:
            raise ValueError("Maximum imbalance must be finite and at least one.")
        self.kind = kind
        self.part_count = parts
        self.halo_width = width
        self.maximum_imbalance = tolerance
        self.curve_depth = depth
        self.graph_seed = seed
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mesh-partition-policy",
                "route": kind.value,
                "part_count": parts,
                "halo_width": width,
                "maximum_imbalance": tolerance,
                "curve_depth": depth,
                "graph_seed": seed,
            }
        )


class MeshPartitionEvidence(StrictModule, NonTrainableState):
    """Measured balance and communication cost of one ownership.

    ``imbalance`` is max part weight over mean part weight, ``edge_cut`` the
    number of face-adjacent cell pairs with different owners, and
    ``halo_replicas`` the total number of ghost copies over all parts.
    """

    part_weights: Array
    imbalance: Array
    edge_cut: Array
    halo_replicas: Array
    kind: MeshPartitionKind = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        owner: np.ndarray,
        weights: np.ndarray,
        adjacency: CellAdjacency,
        halo_replicas: int,
        part_count: int,
        kind: MeshPartitionKind,
        provenance: str,
        /,
    ):
        part_weights = np.bincount(owner, weights=weights, minlength=part_count)
        imbalance = np.max(part_weights) / np.mean(part_weights)
        pairs = adjacency.undirected_pairs()
        edge_cut = np.count_nonzero(owner[pairs[:, 0]] != owner[pairs[:, 1]])
        self.part_weights = jnp.asarray(part_weights, dtype=jnp.float64)
        self.imbalance = jnp.asarray(imbalance, dtype=jnp.float64)
        self.edge_cut = jnp.asarray(edge_cut, dtype=jnp.int64)
        self.halo_replicas = jnp.asarray(halo_replicas, dtype=jnp.int64)
        self.kind = kind
        self.provenance = provenance
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "mesh-partition-evidence",
                "route": kind.value,
                "provenance": provenance,
                "part_weights": array_tree_fingerprint(part_weights),
                "edge_cut": edge_cut,
                "halo_replicas": halo_replicas,
            }
        )


def _native_cell_ids(part: MeshPart) -> np.ndarray:
    carrier = part.carrier
    if isinstance(carrier, CellMeshingResult):
        return np.concatenate(
            tuple(np.asarray(block.global_ids) for block in carrier.mesh.blocks)
        ).astype(np.int64)
    if isinstance(carrier, PreparedTensorGrid):
        return np.arange(prod(carrier.cells().shape), dtype=np.int64)
    if isinstance(carrier, PointCloudPlan):
        return np.arange(carrier.points.shape[0], dtype=np.int64)
    return np.arange(carrier.topology.cell_count, dtype=np.int64)


def _rows_of(native: np.ndarray, identifiers: ArrayLike, /) -> np.ndarray:
    """Native rows of ``identifiers``; -1 where an ID is not a native cell."""
    values = np.asarray(identifiers, dtype=np.int64).reshape(-1)
    sorter = np.argsort(native, kind="stable")
    located = np.minimum(np.searchsorted(native, values, sorter=sorter), native.size - 1)
    rows = sorter[located]
    return np.where(native[rows] == values, rows, -1)


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


def _structured_adjacency(
    shape: tuple[int, ...], periodic: tuple[bool, ...], /
) -> CellAdjacency:
    cells = np.arange(prod(shape), dtype=np.int64).reshape(shape)
    pairs = []
    for axis, (size, wrap) in enumerate(zip(shape, periodic, strict=True)):
        if wrap and size > 1:
            following = np.roll(cells, -1, axis=axis)
            pairs.append(np.stack((cells.reshape(-1), following.reshape(-1)), axis=1))
        else:
            lower = np.take(cells, np.arange(size - 1), axis=axis).reshape(-1)
            upper = np.take(cells, np.arange(1, size), axis=axis).reshape(-1)
            pairs.append(np.stack((lower, upper), axis=1))
    return CellAdjacency(np.concatenate(pairs), cells.size)


def _carrier_adjacency(part: MeshPart, /) -> CellAdjacency:
    carrier = part.carrier
    match carrier:
        case CellMeshingResult():
            return mesh_cell_adjacency(carrier.mesh)
        case PreparedTensorGrid():
            return _structured_adjacency(
                carrier.cells().shape,
                tuple(axis.periodic for axis in carrier.structured_axes),
            )
        case IsogeometricPlan():
            shape = carrier.topology.span_shape
            return _structured_adjacency(shape, (False,) * len(shape))
        case PointCloudPlan():
            # Point stencils are provider-defined; residency must be explicit.
            return CellAdjacency(
                np.zeros((0, 2), dtype=np.int64), carrier.points.shape[0]
            )
        case _:
            raise TypeError("Unsupported mesh carrier.")


def _cell_weights(value: ArrayLike | None, count: int, /) -> np.ndarray:
    if value is None:
        return np.ones((count,), dtype=np.float64)
    weights = np.asarray(value, dtype=np.float64)
    if (
        weights.shape != (count,)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("Cell weights must be finite, positive, one per cell.")
    return weights


def _supplied_halos(
    halos: tuple[ArrayLike, ...],
    native: np.ndarray,
    owner: np.ndarray,
    reach: CellPartitionHalo,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate explicit residency against the required reach; (parts, rows)."""
    values = tuple(np.asarray(value) for value in halos)
    if len(values) != reach.part_count:
        raise ValueError("Exactly one halo residency vector is required per partition.")
    if any(
        value.ndim != 1 or (value.size and not np.issubdtype(value.dtype, np.integer))
        for value in values
    ):
        raise ValueError("Halo residency requires unique native global IDs.")
    sizes = np.asarray([value.size for value in values], dtype=np.int64)
    parts = np.repeat(np.arange(reach.part_count, dtype=np.int64), sizes)
    identifiers = np.concatenate(
        tuple(value.astype(np.int64) for value in values) + (np.zeros((0,), np.int64),)
    )
    rows = _rows_of(native, identifiers)
    keys = parts * native.size + rows
    if np.any(rows < 0) or np.unique(keys).size != keys.size:
        raise ValueError("Halo residency requires unique native global IDs.")
    if np.any(owner[rows] == parts):
        raise ValueError("Halo residency cannot include locally owned entities.")
    halo_offsets = np.asarray(reach.halo_offsets)
    required = np.repeat(
        np.arange(reach.part_count, dtype=np.int64), np.diff(halo_offsets)
    ) * native.size + np.asarray(reach.halo_cells, dtype=np.int64)
    if not np.all(np.isin(required, keys)):
        raise ValueError("Halo residency omits the required native adjacency reach.")
    order = np.lexsort((identifiers, parts))
    return parts[order], rows[order]


def _split_rows(
    parts: np.ndarray, rows: np.ndarray, part_count: int, /
) -> tuple[np.ndarray, ...]:
    offsets = np.cumsum(np.bincount(parts, minlength=part_count))[:-1]
    return tuple(np.split(rows.astype(np.int32), offsets))


def _padded_rows(rows: tuple[Array, ...], /) -> np.ndarray:
    offsets = np.concatenate(([0], np.cumsum([value.size for value in rows])))
    return padded_part_table(
        offsets,
        np.concatenate(
            tuple(np.asarray(value) for value in rows) + (np.zeros((0,), np.int32),)
        ),
    )


class MeshDistribution(StrictModule, NonTrainableState):
    """Revision-bound native ownership and halo residency, independent of a solver.

    Ownership is normalized into the carrier's native cell/span/point ordering.
    Owned and halo rows of every part are sorted by global ID; halo IDs are
    unique and never locally owned. The default halo is the ``halo_width``-layer
    face-adjacency reach of each part. Compact carriers remain intact.
    """

    part: MeshPart
    partition: CellPartition
    cell_global_ids: Array
    cell_weights: Array
    owned_rows: tuple[Array, ...]
    halo_rows: tuple[Array, ...]
    halo_global_ids: tuple[Array, ...]
    dependencies: Array
    evidence: MeshPartitionEvidence
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
        cell_weights: ArrayLike | None = None,
        partition_kind: MeshPartitionKind = MeshPartitionKind.PROVIDER,
        partition_provenance: str = "supplied-ownership",
    ):
        if not isinstance(part, MeshPart) or not isinstance(partition, CellPartition):
            raise TypeError("Mesh distribution requires MeshPart and CellPartition.")
        if not isinstance(partition_kind, MeshPartitionKind):
            raise TypeError("partition_kind must be MeshPartitionKind.")
        width = operator.index(halo_width)
        if isinstance(halo_width, bool) or width < 0:
            raise ValueError("Halo width must be a non-negative integer.")
        native = _native_cell_ids(part)
        ids = native if cell_global_ids is None else np.asarray(cell_global_ids)
        if ids.shape != native.shape or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(
                "Distribution cell IDs must cover the exact native global IDs once."
            )
        position = _rows_of(ids.astype(np.int64), native)
        if np.any(position < 0) or np.unique(ids).size != ids.size:
            raise ValueError(
                "Distribution cell IDs must cover the exact native global IDs once."
            )
        if partition.cell_owner.shape != native.shape:
            raise ValueError("Ownership must cover the exact native cell count.")
        normalized = CellPartition(
            np.asarray(partition.cell_owner)[position], partition.part_count
        )
        owner = np.asarray(normalized.cell_owner)
        weights = _cell_weights(cell_weights, native.size)[position]
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
        adjacency = _carrier_adjacency(part)
        reach = CellPartitionHalo(
            normalized, adjacency, layers=width, cell_global_ids=native
        )
        if halo_global_ids is None:
            halo_parts = np.repeat(
                np.arange(partition.part_count), np.diff(np.asarray(reach.halo_offsets))
            )
            halo_rows = np.asarray(reach.halo_cells, dtype=np.int64)
        else:
            halo_parts, halo_rows = _supplied_halos(halo_global_ids, native, owner, reach)
        owned_parts = owner[np.asarray(reach.owned_cells)]
        owned_rows = _split_rows(
            owned_parts, np.asarray(reach.owned_cells), partition.part_count
        )
        split_halos = _split_rows(halo_parts, halo_rows, partition.part_count)
        dependencies = np.zeros(
            (partition.part_count, partition.part_count), dtype=np.bool_
        )
        dependencies[halo_parts, owner[halo_rows]] = True
        evidence = MeshPartitionEvidence(
            owner,
            weights,
            adjacency,
            halo_rows.size,
            partition.part_count,
            partition_kind,
            str(partition_provenance),
        )
        halo_ids = tuple(native[rows] for rows in split_halos)
        self.part = part
        self.partition = normalized
        self.cell_global_ids = jnp.asarray(native, dtype=jnp.int64)
        self.cell_weights = jnp.asarray(weights, dtype=jnp.float64)
        self.owned_rows = tuple(jnp.asarray(rows) for rows in owned_rows)
        self.halo_rows = tuple(jnp.asarray(rows) for rows in split_halos)
        self.halo_global_ids = tuple(
            jnp.asarray(value, dtype=jnp.int64) for value in halo_ids
        )
        self.dependencies = jnp.asarray(dependencies)
        self.evidence = evidence
        self.split_factors = splits
        self.halo_width = width
        self.distribution_id = canonical_fingerprint(
            {
                "kind": "mesh-distribution",
                "part": part.part_id,
                "partition": normalized.partition_id,
                "cell_ids": array_tree_fingerprint(native),
                "halos": array_tree_fingerprint(halo_ids),
                "split_factors": splits,
                "halo_width": width,
                "evidence": evidence.evidence_id,
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
        owned = _padded_rows(self.owned_rows)
        halo = _padded_rows(self.halo_rows)
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


def _cell_centroids(part: MeshPart, /) -> np.ndarray:
    carrier = part.carrier
    match carrier:
        case CellMeshingResult():
            coordinates = np.asarray(carrier.mesh.coordinates, dtype=np.float64)
            centroids = []
            for block in carrier.mesh.blocks:
                valid = np.asarray(block.vertex_valid, dtype=np.bool_)
                vertices = np.where(valid, np.asarray(block.vertices), 0)
                total = np.sum(
                    np.where(valid[..., None], coordinates[vertices], 0.0), axis=1
                )
                centroids.append(total / np.sum(valid, axis=1)[:, None])
            return np.concatenate(centroids)
        case PreparedTensorGrid():
            centers = np.meshgrid(
                *(
                    np.asarray(axis.interval_centers, dtype=np.float64)
                    for axis in carrier.structured_axes
                ),
                indexing="ij",
            )
            return np.stack(tuple(value.reshape(-1) for value in centers), axis=-1)
        case PointCloudPlan():
            return np.asarray(carrier.points, dtype=np.float64)
        case _:
            raise TypeError(
                "Space-filling-curve partitions require cell, tensor, or point carriers."
            )


def _curve_codes(
    centroids: np.ndarray, kind: MeshPartitionKind, depth: int, /
) -> np.ndarray:
    if not np.all(np.isfinite(centroids)):
        raise ValueError("Space-filling-curve partitions require finite centroids.")
    lower = np.min(centroids, axis=0)
    extent = np.max(np.max(centroids, axis=0) - lower)
    # One isotropic scale keeps the curve's locality in physical distance.
    resolution = (1 << depth) - 1
    scale = resolution / extent if extent > 0.0 else 0.0
    integer = np.clip(np.floor((centroids - lower) * scale), 0, resolution).astype(
        np.int64
    )
    match kind:
        case MeshPartitionKind.MORTON:
            codes = morton_encode_integer(jnp.asarray(integer), depth)
        case MeshPartitionKind.HILBERT:
            codes = hilbert_encode_integer(jnp.asarray(integer), depth)
        case _:
            raise ValueError("Curve codes require the MORTON or HILBERT route.")
    return np.asarray(codes)


def _contiguous_ranges(
    order: np.ndarray, weights: np.ndarray, part_count: int, /
) -> np.ndarray:
    """Split a curve order into weighted contiguous ranges, every part non-empty.

    A cell joins the part whose ideal weight interval contains its weight
    midpoint, so each part deviates from total/part_count by at most one cell
    weight; cut indices are then made strictly increasing so no part is empty.
    """
    count = order.size
    if part_count > count:
        raise ValueError("part_count exceeds the number of cells.")
    ordered = weights[order]
    prefix = np.cumsum(ordered)
    midpoint = part_count * (prefix - 0.5 * ordered) / prefix[-1]
    ideal = np.minimum(np.floor(midpoint).astype(np.int64), part_count - 1)
    levels = np.arange(1, part_count, dtype=np.int64)
    cuts = np.searchsorted(ideal, levels, side="left")
    cuts = np.maximum.accumulate(np.maximum(cuts - levels, 0)) + levels
    cuts = np.minimum(cuts, count - part_count + levels)
    owner = np.empty((count,), dtype=np.int32)
    owner[order] = np.searchsorted(cuts, np.arange(count), side="right")
    return owner


def _partition_owners(
    part: MeshPart,
    weights: np.ndarray,
    policy: MeshPartitionPolicy,
    ownership: ArrayLike | None,
    /,
) -> tuple[np.ndarray, str]:
    if ownership is not None and policy.kind is not MeshPartitionKind.PROVIDER:
        raise ValueError("Supplied ownership requires the PROVIDER partition route.")
    match policy.kind:
        case MeshPartitionKind.MORTON | MeshPartitionKind.HILBERT:
            codes = _curve_codes(_cell_centroids(part), policy.kind, policy.curve_depth)
            order = np.lexsort((_native_cell_ids(part), codes))
            owner = _contiguous_ranges(order, weights, policy.part_count)
            return owner, f"{policy.kind.value}-curve-weighted-ranges"
        case MeshPartitionKind.GRAPH:
            if policy.part_count > weights.size:
                raise ValueError("part_count exceeds the number of cells.")
            adjacency = _carrier_adjacency(part)
            owner = metis_partition(
                np.asarray(adjacency.offsets),
                np.asarray(adjacency.neighbors),
                weights,
                policy.part_count,
                imbalance=policy.maximum_imbalance,
                seed=policy.graph_seed,
            )
            if np.unique(owner).size != policy.part_count:
                raise MetisPartitionError("METIS left at least one part empty.")
            return owner, metis_identity()
        case MeshPartitionKind.PROVIDER:
            if ownership is None:
                raise ValueError("The PROVIDER partition route requires ownership.")
            owner = np.asarray(ownership)
            if owner.shape != weights.shape:
                raise ValueError("Provider ownership must cover every native cell.")
            return owner, "provider-ownership"
        case _:
            raise ValueError("Unsupported mesh partition kind.")


def prepare_mesh_distribution(
    part: MeshPart,
    /,
    *,
    policy: MeshPartitionPolicy,
    cell_weights: ArrayLike | None = None,
    ownership: ArrayLike | None = None,
) -> MeshDistribution:
    """Partition one part by ``policy`` and build its ghost layers.

    ``cell_weights`` and provider ``ownership`` follow native cell order.
    GRAPH raises :class:`MetisUnavailableError` when METIS cannot be loaded.
    """
    if not isinstance(part, MeshPart):
        raise TypeError("part must be MeshPart.")
    if not isinstance(policy, MeshPartitionPolicy):
        raise TypeError("policy must be MeshPartitionPolicy.")
    weights = _cell_weights(cell_weights, _native_cell_ids(part).size)
    owner, provenance = _partition_owners(part, weights, policy, ownership)
    return MeshDistribution(
        part,
        CellPartition(owner, policy.part_count),
        halo_width=policy.halo_width,
        cell_weights=weights,
        partition_kind=policy.kind,
        partition_provenance=provenance,
    )


def _lineage_routes(
    source: MeshDistribution, target_part: MeshPart, lineage: MeshLineage, /
) -> tuple[np.ndarray, np.ndarray]:
    """Validated (source rows, target rows) cell routes covering both meshes."""
    if not isinstance(source, MeshDistribution) or not isinstance(target_part, MeshPart):
        raise TypeError("Transitions require a MeshDistribution and a target MeshPart.")
    if not isinstance(lineage, MeshLineage):
        raise TypeError("lineage must be MeshLineage.")
    if not isinstance(source.part.carrier, CellMeshingResult) or not isinstance(
        target_part.carrier, CellMeshingResult
    ):
        raise TypeError("Distribution transitions require certified cell parts.")
    if target_part.name != source.part.name:
        raise ValueError("Distribution transitions must stay on one named part.")
    source_mesh, target_mesh = source.part.carrier.mesh, target_part.carrier.mesh
    if (
        lineage.source_topology_id != source_mesh.topology_id
        or lineage.target_topology_id != target_mesh.topology_id
    ):
        raise ValueError(
            "Mesh lineage endpoints do not match the source distribution and target part."
        )
    dimension = source_mesh.topological_dimension
    records = tuple(value for value in lineage.entities if value.dimension == dimension)
    if target_mesh.topological_dimension != dimension or not records:
        raise ValueError("Mesh lineage lacks a cell-dimension record.")
    cells = records[0]
    if (
        cells.source_entity_set_id != source_mesh.entity_set(dimension).entity_set_id
        or cells.target_entity_set_id != target_mesh.entity_set(dimension).entity_set_id
    ):
        raise ValueError("Cell lineage entity sets do not match the meshes.")
    source_ids = np.asarray(source.cell_global_ids)
    target_ids = _native_cell_ids(target_part)
    source_rows = _rows_of(source_ids, cells.source_global_ids)
    target_rows = _rows_of(target_ids, cells.target_global_ids)
    created = _rows_of(target_ids, cells.created_target_ids)
    deleted = _rows_of(source_ids, cells.deleted_source_ids)
    if np.any(source_rows < 0) or np.any(deleted < 0):
        raise ValueError("Cell lineage references IDs outside the source mesh.")
    if np.any(target_rows < 0) or np.any(created < 0):
        raise ValueError("Cell lineage references IDs outside the target mesh.")
    if not np.all(np.isin(np.arange(target_ids.size), np.union1d(target_rows, created))):
        raise ValueError("Cell lineage must account for every target cell.")
    if not np.all(np.isin(np.arange(source_ids.size), np.union1d(source_rows, deleted))):
        raise ValueError("Cell lineage must account for every source cell.")
    return source_rows, target_rows


def _grow_ownership(owner: np.ndarray, adjacency: CellAdjacency, /) -> np.ndarray:
    """Give unowned cells the majority owner of owned face neighbors, layer by layer."""
    pairs = adjacency.undirected_pairs().astype(np.int64)
    first = np.concatenate((pairs[:, 0], pairs[:, 1]))
    second = np.concatenate((pairs[:, 1], pairs[:, 0]))
    grown = owner.copy()
    while True:
        frontier = (grown[second] < 0) & (grown[first] >= 0)
        if not np.any(frontier):
            return grown
        proposed, _ = inherit_cell_owners(
            grown, first[frontier], second[frontier], grown.size
        )
        grown = np.where(grown < 0, proposed, grown)


def _relabel_to_inherited(
    fresh: np.ndarray, inherited: np.ndarray, weights: np.ndarray, part_count: int, /
) -> np.ndarray:
    """Rename fresh parts to the inherited ranks they overlap most (assignment)."""
    # Imported lazily: the transport package is heavy and unrelated to meshing import.
    from ..transport import solve_multidimensional_assignment

    defined = inherited >= 0
    overlap = np.bincount(
        fresh[defined] * part_count + inherited[defined],
        weights=weights[defined],
        minlength=part_count * part_count,
    ).reshape(part_count, part_count)
    anchors = np.zeros((part_count, 1), dtype=np.float64)
    assignment = solve_multidimensional_assignment(
        anchors,
        anchors,
        cost=np.max(overlap) - overlap,
        maximum_atoms=part_count,
    )
    mapping = np.empty((part_count,), dtype=np.int32)
    mapping[np.asarray(assignment.source_indices)] = np.asarray(assignment.target_indices)
    return mapping[fresh]


def _transition_owners(
    source: MeshDistribution,
    target_part: MeshPart,
    routes: tuple[np.ndarray, np.ndarray],
    weights: np.ndarray,
    policy: MeshPartitionPolicy,
    ownership: ArrayLike | None,
    /,
) -> tuple[np.ndarray, bool, str]:
    """Keep inherited owners when balanced; otherwise repartition and remap ranks."""
    parts = policy.part_count
    if policy.kind is MeshPartitionKind.PROVIDER:
        owner, provenance = _partition_owners(target_part, weights, policy, ownership)
        return owner, True, f"transition-{provenance}"
    inherited, _ = inherit_cell_owners(
        np.asarray(source.partition.cell_owner), routes[0], routes[1], weights.size
    )
    grown = _grow_ownership(inherited, _carrier_adjacency(target_part))
    if np.all(grown >= 0):
        part_weights = np.bincount(grown, weights=weights, minlength=parts)
        if (
            np.all(part_weights > 0.0)
            and np.max(part_weights) / np.mean(part_weights) <= policy.maximum_imbalance
        ):
            return grown, False, "transition-inherited"
    fresh, provenance = _partition_owners(target_part, weights, policy, ownership)
    relabeled = _relabel_to_inherited(fresh, inherited, weights, parts)
    return relabeled, True, f"transition-rebalanced:{provenance}"


class MeshDistributionTransition(StrictModule, NonTrainableState):
    """Ownership carried through mesh lineage onto an adapted part revision.

    Each lineage route is one message slot; slots are ordered by (source rank,
    target rank, target global ID, source global ID). ``send`` gathers source
    cells into slots and ``receive`` reduces slots onto target cells, so
    :meth:`transfer` moves cell data exactly as rank-to-rank migration would.
    ``message_offsets`` groups slots by the source-rank-major pair
    ``source_rank * part_count + target_rank``. Global IDs are never renumbered.
    """

    source: MeshDistribution
    target: MeshDistribution
    send: EdgeRelation
    receive: EdgeRelation
    message_source_ranks: Array
    message_target_ranks: Array
    message_offsets: Array
    migration_counts: Array
    target_defined: Array
    migrated_cells: Array
    migration_volume: Array
    rebalanced: bool = eqx.field(static=True)
    lineage_id: str = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: MeshDistribution,
        target: MeshDistribution,
        source_rows: ArrayLike,
        target_rows: ArrayLike,
        lineage: MeshLineage,
        /,
        *,
        rebalanced: bool,
    ):
        if not isinstance(source, MeshDistribution) or not isinstance(
            target, MeshDistribution
        ):
            raise TypeError("Transition endpoints must be MeshDistribution values.")
        if not isinstance(lineage, MeshLineage):
            raise TypeError("lineage must be MeshLineage.")
        parts = source.partition.part_count
        if target.partition.part_count != parts:
            raise ValueError("Distribution transitions keep the part count.")
        sources = np.asarray(source_rows, dtype=np.int64)
        targets = np.asarray(target_rows, dtype=np.int64)
        source_count = source.cell_global_ids.size
        target_count = target.cell_global_ids.size
        if (
            sources.ndim != 1
            or targets.shape != sources.shape
            or np.any((sources < 0) | (sources >= source_count))
            or np.any((targets < 0) | (targets >= target_count))
        ):
            raise ValueError("Transition routes must address native source/target rows.")
        sender = np.asarray(source.partition.cell_owner, dtype=np.int64)[sources]
        receiver = np.asarray(target.partition.cell_owner, dtype=np.int64)[targets]
        order = np.lexsort(
            (
                np.asarray(source.cell_global_ids)[sources],
                np.asarray(target.cell_global_ids)[targets],
                receiver,
                sender,
            )
        )
        sources, targets = sources[order], targets[order]
        sender, receiver = sender[order], receiver[order]
        slots = np.arange(sources.size, dtype=np.int32)
        pair_counts = np.bincount(sender * parts + receiver, minlength=parts * parts)
        offsets = np.concatenate(([0], np.cumsum(pair_counts)))
        remote = np.unique(targets[sender != receiver])
        volume = np.sum(np.asarray(target.cell_weights)[remote])
        self.source = source
        self.target = target
        self.send = EdgeRelation(
            sources, slots, source_size=source_count, target_size=sources.size
        )
        self.receive = EdgeRelation(
            slots, targets, source_size=sources.size, target_size=target_count
        )
        self.message_source_ranks = jnp.asarray(sender, dtype=jnp.int32)
        self.message_target_ranks = jnp.asarray(receiver, dtype=jnp.int32)
        self.message_offsets = jnp.asarray(offsets, dtype=jnp.int64)
        self.migration_counts = jnp.asarray(
            pair_counts.reshape(parts, parts), dtype=jnp.int64
        )
        self.target_defined = jnp.asarray(
            np.bincount(targets, minlength=target_count) > 0
        )
        self.migrated_cells = jnp.asarray(remote.size, dtype=jnp.int64)
        self.migration_volume = jnp.asarray(volume, dtype=jnp.float64)
        self.rebalanced = bool(rebalanced)
        self.lineage_id = lineage.lineage_id
        self.transition_id = canonical_fingerprint(
            {
                "kind": "mesh-distribution-transition",
                "source": source.distribution_id,
                "target": target.distribution_id,
                "lineage": lineage.lineage_id,
                "sources": array_tree_fingerprint(sources),
                "targets": array_tree_fingerprint(targets),
                "rebalanced": self.rebalanced,
            }
        )

    def _rank(self, rank: int, /) -> int:
        index = operator.index(rank)
        if not 0 <= index < self.source.partition.part_count:
            raise ValueError("Transition rank is out of range.")
        return index

    def sent_by(self, rank: int, /) -> Array:
        """Message slots sent by ``rank``, grouped by receiving rank."""
        index = self._rank(rank)
        parts = self.source.partition.part_count
        offsets = np.asarray(self.message_offsets)
        return jnp.arange(offsets[index * parts], offsets[(index + 1) * parts])

    def received_by(self, rank: int, /) -> Array:
        """Message slots received by ``rank``, grouped by sending rank."""
        index = self._rank(rank)
        parts = self.source.partition.part_count
        offsets = np.asarray(self.message_offsets)
        pairs = np.arange(parts) * parts + index
        starts, stops = offsets[pairs], offsets[pairs + 1]
        counts = stops - starts
        shift = np.repeat(starts - (np.cumsum(counts) - counts), counts)
        return jnp.asarray(np.arange(np.sum(counts)) + shift, dtype=jnp.int32)

    def transfer(
        self, source_values: ArrayLike, /, *, reduction: RouteReduction = "mean"
    ) -> Array:
        """Migrate native-order source cell data onto native target cells.

        Preserved and refined cells copy their parent value; merged cells reduce
        their sources with ``reduction``. Created cells (``~target_defined``)
        receive zero and must be initialized by the consumer.
        """
        values = jnp.asarray(source_values)
        if values.ndim == 0 or values.shape[0] != self.source.cell_global_ids.size:
            raise ValueError("Transferred values must follow source native cell order.")
        return route_reduce(
            self.receive, gather_routes(self.send, values), reduction=reduction
        )


def prepare_distribution_transition(
    source: MeshDistribution,
    target_part: MeshPart,
    lineage: MeshLineage,
    /,
    *,
    policy: MeshPartitionPolicy,
    cell_weights: ArrayLike | None = None,
    ownership: ArrayLike | None = None,
) -> MeshDistributionTransition:
    """Carry ownership through ``lineage`` onto ``target_part`` and rebuild ghosts.

    Refined and preserved cells inherit their parent's owner, merged cells the
    majority owner, and created cells the majority owner of owned neighbors.
    If that ownership exceeds ``policy.maximum_imbalance`` (or leaves a part
    empty) the target is repartitioned by ``policy`` and parts are renamed to
    the source ranks they overlap most, minimizing migration. The PROVIDER
    route takes target ``ownership`` as supplied.
    """
    if not isinstance(policy, MeshPartitionPolicy):
        raise TypeError("policy must be MeshPartitionPolicy.")
    if ownership is not None and policy.kind is not MeshPartitionKind.PROVIDER:
        raise ValueError("Supplied ownership requires the PROVIDER partition route.")
    routes = _lineage_routes(source, target_part, lineage)
    if policy.part_count != source.partition.part_count:
        raise ValueError("Distribution transitions keep the source part count.")
    weights = _cell_weights(cell_weights, _native_cell_ids(target_part).size)
    owner, rebalanced, provenance = _transition_owners(
        source, target_part, routes, weights, policy, ownership
    )
    target = MeshDistribution(
        target_part,
        CellPartition(owner, policy.part_count),
        halo_width=policy.halo_width,
        cell_weights=weights,
        partition_kind=policy.kind,
        partition_provenance=provenance,
    )
    return MeshDistributionTransition(
        source, target, routes[0], routes[1], lineage, rebalanced=rebalanced
    )


__all__ = [
    "MeshDistribution",
    "MeshDistributionTransition",
    "MeshPartitionEvidence",
    "MeshPartitionKind",
    "MeshPartitionPolicy",
    "prepare_distribution_transition",
    "prepare_mesh_distribution",
]
