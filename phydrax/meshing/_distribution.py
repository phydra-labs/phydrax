#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Mesh ownership, ghost residency, and ownership migration across revisions.

Ownership comes from weighted contiguous ranges along Morton or Hilbert
space-filling curves, from the native multilevel graph partitioner of
:mod:`phydrax.graph`, from the explicitly selected METIS comparison plugin, or
from a provider. Ghost layers are breadth-first face-adjacency
reach. Transitions carry ownership through mesh lineage onto adapted parts,
repartition only when inherited ownership is out of balance, and migrate cell
data through explicit send/receive sparse relations.
"""

from __future__ import annotations

import operator
from enum import StrEnum
from math import prod
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import PartitionSpec
from jax.typing import ArrayLike

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    logical_array_value_collection_digest,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellPartition, PointCloudPlan, PreparedTensorGrid
from ..discretization._adaptive_simplex import (
    _children,
    _facet_columns,
    _members,
    _refinement_edges,
    _restored_facet_classes,
    _terminal,
    _work,
    AdaptiveSimplexParts,
    AdaptiveSimplexState,
    AdaptiveSimplexStatus,
)
from ..discretization._cell_mesh import CellMesh
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
from ..graph._partition import GraphPartitionPlan, partition_graph, WeightedCSRGraph
from ..sparse import EdgeRelation, gather_routes, route_reduce, RouteReduction
from ..typing import checked
from ._assembly import MeshPart
from ._lineage import MeshLineage
from ._metis import metis_identity, metis_partition, MetisPartitionError
from ._result import (
    CellMeshingResult,
    CollectiveMeshEvidence,
    require_original_meshing_source,
)


if TYPE_CHECKING:
    from ..discretization.fem._distributed import (
        FiniteElementClosurePreparation,
        FiniteElementGlobalDofOwnership,
        OwnerLocalFiniteElementDiscretization,
    )
    from ..discretization.fem._generic import FiniteElementPlan
    from ..discretization.fem._topology_transfer import (
        FiniteElementFieldTransfer,
    )
    from ..lifecycle import CompositionEntry, CompositionTransport
    from ._bisection import BisectionUniformRefinement
    from ._contracts import MeshingLimits


_MAXIMUM_CURVE_DEPTH = 21


class MeshPartitionKind(StrEnum):
    """Ownership route: space-filling curve, graph partitioner, or provider.

    ``GRAPH`` is the native deterministic multilevel partitioner; ``METIS`` is
    the explicit external comparison route and never substitutes for it.
    """

    MORTON = "morton"
    HILBERT = "hilbert"
    GRAPH = "graph"
    METIS = "metis"
    PROVIDER = "provider"


class MeshPartitionPolicy(StrictModule, NonTrainableState):
    """How cell ownership is produced, balanced, and ghosted.

    ``maximum_imbalance`` bounds max-part-weight / mean-part-weight: it is the
    GRAPH and METIS balance target and the tolerance under which a transition
    keeps inherited ownership instead of repartitioning. Curve routes guarantee
    ``imbalance <= 1 + part_count * max_cell_weight / total_weight`` whenever
    every part can be non-empty. ``curve_depth`` is the per-axis bit depth of
    the isotropic curve quantization; equal codes order by global ID.
    ``metis_seed`` seeds the METIS comparison route only; the native GRAPH
    route is deterministic without a seed.
    """

    kind: MeshPartitionKind = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    halo_width: int = eqx.field(static=True)
    maximum_imbalance: float = eqx.field(static=True)
    curve_depth: int = eqx.field(static=True)
    metis_seed: int = eqx.field(static=True)
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
        metis_seed: int = 0,
    ) -> None:
        if not isinstance(kind, MeshPartitionKind):
            raise TypeError("kind must be MeshPartitionKind.")
        counts = (part_count, halo_width, curve_depth, metis_seed)
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
        self.metis_seed = seed
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mesh-partition-policy",
                "route": kind.value,
                "part_count": parts,
                "halo_width": width,
                "maximum_imbalance": tolerance,
                "curve_depth": depth,
                "metis_seed": seed,
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
        *,
        _collective_summary: tuple[Array, Array, Array] | None = None,
    ) -> None:
        if _collective_summary is None:
            part_weights = np.bincount(owner, weights=weights, minlength=part_count)
            pairs = adjacency.undirected_pairs()
            edge_cut = np.count_nonzero(owner[pairs[:, 0]] != owner[pairs[:, 1]])
        else:
            part_weights = np.asarray(
                jax.device_get(_collective_summary[0]), dtype=np.float64
            )
            edge_cut = np.asarray(
                jax.device_get(_collective_summary[1]), dtype=np.int64
            ).item()
            halo_replicas = np.asarray(
                jax.device_get(_collective_summary[2]), dtype=np.int64
            ).item()
            if (
                part_weights.shape != (part_count,)
                or not np.all(np.isfinite(part_weights))
                or np.any(part_weights <= 0)
                or edge_cut < 0
                or halo_replicas < 0
            ):
                raise ValueError(
                    "Collective partition summaries require positive finite part weights."
                )
        imbalance = np.max(part_weights) / np.mean(part_weights)
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
    local_scope: bool = eqx.field(static=True)
    global_cell_count: int = eqx.field(static=True)
    cell_content_id: str = eqx.field(static=True)

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
        _collective_summary: tuple[Array, Array, Array] | None = None,
    ) -> None:
        if not isinstance(part, MeshPart) or not isinstance(partition, CellPartition):
            raise TypeError("Mesh distribution requires MeshPart and CellPartition.")
        if not isinstance(partition_kind, MeshPartitionKind):
            raise TypeError("partition_kind must be MeshPartitionKind.")
        width = operator.index(halo_width)
        if isinstance(halo_width, bool) or width < 0:
            raise ValueError("Halo width must be a non-negative integer.")
        if (
            isinstance(part.carrier, CellMeshingResult)
            and part.carrier.mesh.storage is not None
        ):
            storage = part.carrier.mesh.storage
            native = np.asarray(storage.entity_global_ids[-1], dtype=np.int64)
            owner = np.asarray(partition.cell_owner, dtype=np.int32)
            if (
                partition.part_count != storage.partition_count
                or partition.storage_id != storage.storage_id
                or not np.array_equal(owner, np.asarray(storage.entity_owner[-1]))
                or width > storage.neighborhood_depth
                or split_factors is not None
                or cell_global_ids is not None
                or halo_global_ids is not None
                or _collective_summary is None
            ):
                raise ValueError(
                    "Owner-local distribution requires consumed ownership/halo and collective summaries."
                )
            collective = part.carrier.collective_evidence
            if collective is None:
                raise ValueError(
                    "Owner-local distribution requires consumed collective mesh evidence."
                )
            collective.require_passed()
            rank = storage.partition_index
            weights = _cell_weights(cell_weights, native.size)
            owned = np.flatnonzero(owner == rank)
            ghosts = np.flatnonzero(owner != rank)
            owned = owned[np.argsort(native[owned], kind="stable")]
            ghosts = ghosts[np.argsort(native[ghosts], kind="stable")]
            empty = jnp.empty((0,), dtype=jnp.int32)
            self.part = part
            self.partition = partition
            self.cell_global_ids = storage.entity_global_ids[-1]
            self.cell_weights = jnp.asarray(weights, dtype=jnp.float64)
            self.owned_rows = tuple(
                jnp.asarray(owned, dtype=jnp.int32) if index == rank else empty
                for index in range(storage.partition_count)
            )
            self.halo_rows = tuple(
                jnp.asarray(ghosts, dtype=jnp.int32) if index == rank else empty
                for index in range(storage.partition_count)
            )
            self.halo_global_ids = tuple(
                self.cell_global_ids[rows] for rows in self.halo_rows
            )
            dependencies = np.zeros(
                (storage.partition_count, storage.partition_count), dtype=np.bool_
            )
            dependencies[rank, owner[ghosts]] = True
            self.dependencies = jnp.asarray(dependencies, dtype=jnp.bool_)
            self.evidence = MeshPartitionEvidence(
                owner,
                weights,
                _carrier_adjacency(part),
                ghosts.size,
                storage.partition_count,
                partition_kind,
                partition_provenance,
                _collective_summary=_collective_summary,
            )
            self.split_factors = None
            self.halo_width = width
            self.local_scope = True
            self.global_cell_count = storage.global_entity_counts[-1]
            self.cell_content_id = canonical_fingerprint(array_tree_fingerprint(native))
            self.distribution_id = canonical_fingerprint(
                {
                    "kind": "owner-local-mesh-distribution",
                    "part": part.part_id,
                    "storage": storage.storage_id,
                    "collective": collective.evidence_id,
                    "partition_evidence": self.evidence.evidence_id,
                    "halo_width": width,
                }
            )
            return
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
        self.local_scope = False
        self.global_cell_count = native.size
        self.cell_content_id = canonical_fingerprint(array_tree_fingerprint(native))
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

    def prepare_finite_element_closures(
        self,
        source_plan: FiniteElementPlan,
        /,
        *,
        limits: MeshingLimits,
        cell_capacity: int,
        vertex_capacity: int,
        message_capacity: int,
        axis_name: str,
        source_discretization: FiniteElementDiscretization | None = None,
        transfer: FiniteElementFieldTransfer | None = None,
        target_authority: FiniteElementGlobalDofOwnership | None = None,
        numeric_version: str = "0",
    ) -> FiniteElementClosurePreparation:
        """Prepare genuine source-owned FE closures before any local numbering.

        The immutable source remains the scientific authority. Communication
        support is resolved to its actual cells before closure allocation.
        This does not run a global numerical operator or a periodic device edit.
        """
        from ..discretization.fem._distributed import FiniteElementClosurePreparation
        from ._measurements import NativeExecutionRecord
        from ._volume_generation import native_volume_execution_budget

        with native_volume_execution_budget(limits) as budget:
            authority, programs = _prepare_source_finite_element_closures(
                self,
                source_plan,
                limits=limits,
                cell_capacity=cell_capacity,
                vertex_capacity=vertex_capacity,
                message_capacity=message_capacity,
                axis_name=axis_name,
                source_discretization=source_discretization,
                transfer=transfer,
                target_authority=target_authority,
                numeric_version=numeric_version,
            )
        receipt = (
            None
            if budget.evidence is None
            else NativeExecutionRecord(
                budget.evidence,
                owner_id=authority.plan_id,
            )
        )
        return FiniteElementClosurePreparation(
            authority, programs, execution_evidence=receipt
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


def _prepare_source_finite_element_closures(
    distribution: MeshDistribution,
    source_plan: FiniteElementPlan,
    /,
    *,
    limits: MeshingLimits,
    cell_capacity: int,
    vertex_capacity: int,
    message_capacity: int,
    axis_name: str,
    source_discretization: FiniteElementDiscretization | None,
    transfer: FiniteElementFieldTransfer | None,
    target_authority: FiniteElementGlobalDofOwnership | None,
    numeric_version: str,
) -> tuple[
    FiniteElementGlobalDofOwnership,
    tuple[OwnerLocalFiniteElementDiscretization, ...],
]:
    from .._meshcore import charge_native_geometry_queries, current_native_host_workspace
    from ..discretization._cell_geometry import (
        CellGeometryRestrictionSource,
        CellGeometrySpec,
    )
    from ..discretization._cell_geometry_validity import cell_geometry_id
    from ..discretization._cell_mesh import CellBlock, CellMeshStorage
    from ..discretization._coordinate_enclosure import coordinate_source_signature
    from ..discretization._exact_power_geometry import (
        ExactPowerCellGeometryLinearActionSource,
        ExactPowerCellGeometryRestrictionSource,
        ExactPowerCellGeometrySource,
    )
    from ..discretization.fem._distributed import (
        FiniteElementGlobalDofOwnership,
        owner_local_finite_element_transfer_support,
    )
    from ..discretization.fem._generic import FiniteElementPlan
    from ._collective_geometry import (
        periodic_source_closure_cells,
        project_source_periodic_topology,
    )
    from ._topology_edit import entity_keys

    distribution.require_current(distribution.part)
    source = distribution.part.carrier
    if not isinstance(source, CellMeshingResult) or source.certification is None:
        raise ValueError(
            "Native FE closures require an actual certified immutable whole source."
        )
    if not isinstance(source_plan, FiniteElementPlan) or (
        source_plan.mesh.storage is not None
        or array_tree_fingerprint(source_plan.mesh) != array_tree_fingerprint(source.mesh)
        or cell_geometry_id(source_plan.coordinate_spec)
        != cell_geometry_id(source.geometry)
    ):
        raise ValueError(
            "Native FE closures require the actual source FE mesh and geometry."
        )
    mesh, geometry = source.mesh, source_plan.coordinate_spec
    cell_capacity, vertex_capacity = (
        operator.index(cell_capacity),
        operator.index(vertex_capacity),
    )
    if (
        not 0 < cell_capacity <= limits.maximum_cells
        or not 0 < vertex_capacity <= limits.maximum_vertices
    ):
        raise ValueError("FE closure capacities exceed the original entity limits.")
    if any(not isinstance(block, CellBlock) for block in mesh.blocks):
        raise TypeError(
            "Source closure extraction requires canonical fixed-family blocks."
        )
    workspace = current_native_host_workspace()
    if workspace is None:
        raise RuntimeError(
            "Native FE closure preparation requires its original storage workspace."
        )
    workspace.retain_owner(
        (distribution, source_plan, source_discretization, transfer, target_authority)
    )
    charge_native_geometry_queries(
        0, work_units=sum(block.vertices.size for block in mesh.blocks)
    )
    authority = FiniteElementGlobalDofOwnership(
        source_plan, distribution.partition, source_discretization=source_discretization
    )
    workspace.retain_owner(authority)
    if (transfer is None) != (target_authority is None):
        raise ValueError(
            "Transfer support requires the actual transfer and target authority together."
        )
    if transfer is None:
        support = tuple(
            np.zeros(0, dtype=np.int64) for _ in range(distribution.partition.part_count)
        )
    else:
        if target_authority is None:
            raise RuntimeError("Transfer target authority disappeared after validation.")
        support = owner_local_finite_element_transfer_support(
            transfer, authority, target_authority
        )
    if len(support) != distribution.partition.part_count:
        raise ValueError("Transfer support differs from the actual source owner count.")
    cell_ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    cell_lookup = {int(identifier): row for row, identifier in enumerate(cell_ids)}
    closures = []
    for rank, requested in enumerate(support):
        rows = np.concatenate(
            (
                np.asarray(distribution.owned_rows[rank]),
                np.asarray(distribution.halo_rows[rank]),
            )
        )
        required = np.unique(
            np.concatenate(
                (cell_ids[rows], np.asarray(authority.supporting_cells(requested)))
            )
        )
        required = periodic_source_closure_cells(
            mesh, required, cell_capacity=cell_capacity, vertex_capacity=vertex_capacity
        )
        if required.size > cell_capacity:
            raise ValueError(
                "Actual transfer support exceeds the original closure capacity."
            )
        closures.append(
            np.asarray([cell_lookup[int(identifier)] for identifier in required])
        )
    owners = [None] * (mesh.topological_dimension + 1)
    owners[-1] = np.asarray(distribution.partition.cell_owner)
    for degree in range(mesh.topological_dimension, 0, -1):
        relation = mesh.topology.incidences[degree - 1].relation
        valid = np.asarray(relation.valid)
        cell_owner = owners[degree]
        if cell_owner is None:
            raise RuntimeError("A higher-dimensional source owner is absent.")
        entity_owner = np.full(
            mesh.entity_set(degree - 1).count,
            distribution.partition.part_count,
            dtype=np.int32,
        )
        np.minimum.at(
            entity_owner,
            np.asarray(relation.source_indices)[valid],
            cell_owner[np.asarray(relation.target_indices)[valid]],
        )
        if np.any(entity_owner == distribution.partition.part_count):
            raise ValueError("An actual source entity lacks incident-cell ownership.")
        owners[degree - 1] = entity_owner
    entity_owners = tuple(owner for owner in owners if isinstance(owner, np.ndarray))
    if len(entity_owners) != mesh.topological_dimension + 1:
        raise RuntimeError("Source entity ownership preparation is incomplete.")
    elements, routes, _ = geometry.resolve(mesh)
    coordinate_owners = np.full(
        geometry.coordinates.shape[0], distribution.partition.part_count, dtype=np.int32
    )
    arrays = {
        "coordinates": mesh.coordinates,
        "vertex_global_ids": mesh.vertex_global_ids,
        "cell_global_ids": jnp.asarray(cell_ids),
        "cell_owners": jnp.asarray(entity_owners[-1]),
        "geometry/coordinate_ids": jnp.arange(
            geometry.coordinates.shape[0], dtype=jnp.int64
        ),
        "geometry/coordinates": geometry.coordinates,
    }
    origin = geometry.restriction_source
    source_geometry_id = (
        cell_geometry_id(geometry) if origin is None else origin.source_geometry_id
    )
    source_topology_id = mesh.topology_id if origin is None else origin.source_topology_id
    source_blocks = None if origin is None else origin.block_source_blocks
    for name, identity in (
        ("source_geometry_id", source_geometry_id),
        ("source_topology_id", source_topology_id),
    ):
        arrays[f"geometry/{name}"] = jnp.asarray(
            np.frombuffer(bytes.fromhex(canonical_fingerprint(identity)), dtype=np.uint8)
        )
    for degree in range(mesh.topological_dimension + 1):
        arrays[f"entity/{degree}/ids"] = mesh.entity_set(degree).entity_ids
        arrays[f"entity/{degree}/owners"] = jnp.asarray(entity_owners[degree])
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        rows = np.asarray(
            [cell_lookup[int(identifier)] for identifier in np.asarray(block.global_ids)]
        )
        route_values = np.asarray(route).reshape(-1)
        route_owners = np.broadcast_to(
            entity_owners[-1][rows, None], route.shape
        ).reshape(-1)
        valid_routes = route_values >= 0
        np.minimum.at(
            coordinate_owners, route_values[valid_routes], route_owners[valid_routes]
        )
        digest = canonical_fingerprint(
            {
                "signature": coordinate_source_signature(element),
                "arrays": array_tree_fingerprint(element),
            }
        )
        (
            arrays[f"geometry/cell_ids/{block.name}"],
            arrays[f"geometry/routes/{block.name}"],
        ) = block.global_ids, route
        arrays[f"geometry/source_basis/{block.name}"] = jnp.broadcast_to(
            np.frombuffer(bytes.fromhex(digest), dtype=np.uint8), (block.cell_count, 32)
        )
        arrays[f"geometry/action_counts/{block.name}"] = jnp.zeros(
            block.cell_count, dtype=jnp.int32
        )
        arrays[f"geometry/action_weights/{block.name}"] = jnp.zeros(
            (
                block.cell_count,
                0,
                block.topological_dimension + 1,
                block.topological_dimension + 1,
            ),
            dtype=jnp.float64,
        )
        arrays[f"geometry/parent_cell_ids/{block.name}"] = (
            block.global_ids
            if origin is None
            else origin.block_parent_cell_ids[block.name]
        )
        arrays[f"geometry/parent_vertex_ids/{block.name}"] = (
            mesh.vertex_global_ids[block.vertices]
            if origin is None
            else origin.block_parent_vertex_ids[block.name]
        )
    if np.any(coordinate_owners == distribution.partition.part_count):
        raise ValueError("An actual source coefficient lacks incident-cell ownership.")
    arrays["geometry/coordinate_owners"] = jnp.asarray(coordinate_owners)
    retained_bytes = sum(value.nbytes for value in arrays.values())
    if retained_bytes > limits.maximum_scratch_bytes:
        raise ValueError(
            "Actual scientific source banks exceed the original storage allowance."
        )
    workspace.retain_owner((arrays, closures))
    local_plans = []
    offsets = np.cumsum([0] + [block.cell_count for block in mesh.blocks])
    for rank, selected in enumerate(closures):
        charge_native_geometry_queries(
            0, work_units=sum(block.vertices.size for block in mesh.blocks)
        )
        selections = [
            selected[(selected >= left) & (selected < right)] - left
            for left, right in zip(offsets[:-1], offsets[1:], strict=True)
        ]
        active = [
            (block, element, route, rows)
            for block, element, route, rows in zip(
                mesh.blocks, elements, routes, selections, strict=True
            )
            if rows.size
        ]
        vertex_rows = np.unique(
            np.concatenate(
                [
                    np.asarray(block.vertices)[rows].reshape(-1)
                    for block, _, _, rows in active
                ]
            )
        )
        if vertex_rows.size > vertex_capacity:
            raise ValueError(
                "Actual source closure exceeds the original vertex capacity."
            )
        blocks = tuple(
            CellBlock(
                block.name,
                block.cell_kind,
                np.searchsorted(vertex_rows, np.asarray(block.vertices)[rows]),
                global_ids=np.asarray(block.global_ids)[rows],
            )
            for block, _, _, rows in active
        )
        points, vertex_ids = (
            mesh.coordinates[vertex_rows],
            mesh.vertex_global_ids[vertex_rows],
        )
        probe = CellMesh(points, blocks, vertex_global_ids=vertex_ids)
        local_ids, local_owners, intermediate = [], [], {}
        for degree in range(mesh.topological_dimension + 1):
            lookup = {
                tuple(key): row for row, key in enumerate(entity_keys(mesh, degree))
            }
            rows = np.asarray(
                [lookup[tuple(key)] for key in entity_keys(probe, degree)],
                dtype=np.int64,
            )
            identifiers = np.asarray(mesh.entity_set(degree).entity_ids)[rows]
            local_ids.append(identifiers)
            local_owners.append(entity_owners[degree][rows])
            if 0 < degree < mesh.topological_dimension:
                intermediate[degree] = identifiers
        lifted = CellMesh(
            points, blocks, vertex_global_ids=vertex_ids, entity_global_ids=intermediate
        )
        periodic = project_source_periodic_topology(mesh, lifted)
        selected_routes = {
            block.name: np.asarray(route)[rows] for block, _, route, rows in active
        }
        coordinate_ids = np.unique(
            np.concatenate([route[route >= 0] for route in selected_routes.values()])
        )
        local_elements = {block.name: element for block, element, _, _ in active}
        local_routes = {
            name: np.where(
                route >= 0, np.searchsorted(coordinate_ids, np.maximum(route, 0)), -1
            )
            for name, route in selected_routes.items()
        }
        local_origin = CellGeometryRestrictionSource(
            source_geometry_id,
            source_topology_id,
            {
                block.name: arrays[f"geometry/parent_cell_ids/{block.name}"][rows]
                for block, _, _, rows in active
            },
            {
                block.name: arrays[f"geometry/parent_vertex_ids/{block.name}"][rows]
                for block, _, _, rows in active
            },
            block_source_blocks=None
            if source_blocks is None
            else {block.name: source_blocks[block.name] for block, _, _, _ in active},
        )
        local_source = (
            None
            if geometry.periodic_source is None
            else geometry.periodic_source.reindexed(coordinate_ids)
        )
        local_exact_source = geometry.exact_source
        if isinstance(
            local_exact_source,
            (
                ExactPowerCellGeometrySource,
                ExactPowerCellGeometryRestrictionSource,
                ExactPowerCellGeometryLinearActionSource,
            ),
        ):
            local_exact_source = ExactPowerCellGeometryRestrictionSource(
                local_exact_source,
                np.empty((0, 4), dtype=np.float64),
                np.column_stack(
                    (coordinate_ids, np.full(coordinate_ids.size, -1, dtype=np.int64))
                ),
                np.full(coordinate_ids.size, -1, dtype=np.int64),
            )
        local_geometry = CellGeometrySpec(
            local_elements,
            local_routes,
            geometry.coordinates[coordinate_ids],
            restriction_source=local_origin,
            periodic_source=local_source,
            exact_source=local_exact_source,
        )
        workspace.retain_owner(
            (blocks, periodic, local_geometry, local_ids, local_owners)
        )
        retained_bytes += (
            points.nbytes
            + sum(block.vertices.nbytes for block in blocks)
            + local_geometry.coordinates.nbytes
        )
        if retained_bytes > limits.maximum_scratch_bytes:
            raise ValueError(
                "Actual FE closure storage exceeds the original storage allowance."
            )
        storage = CellMeshStorage(
            tuple(
                mesh.entity_set(degree).count
                for degree in range(mesh.topological_dimension + 1)
            ),
            local_ids,
            local_owners,
            partition_index=rank,
            partition_count=distribution.partition.part_count,
            local_coordinates=points,
            local_blocks=blocks,
            logical_topology_id=mesh.topology_id,
            logical_geometry_id=mesh.geometry_id,
            logical_coordinate_geometry_id=cell_geometry_id(geometry),
            evidence_id=source.result_id,
            logical_arrays=tuple(sorted(arrays.items())),
            local_geometry=local_geometry,
            coordinate_global_ids=coordinate_ids,
            coordinate_owner=coordinate_owners[coordinate_ids],
            global_coordinate_count=geometry.coordinates.shape[0],
            neighborhood_depth=distribution.halo_width,
        )
        local_mesh = CellMesh(
            points,
            blocks,
            periodic_topology=periodic,
            storage=storage,
            numeric_version=mesh.numeric_version,
        )
        local_spec = CellGeometrySpec(
            local_elements,
            local_routes,
            local_geometry.coordinates,
            storage=storage,
            restriction_source=local_origin,
            periodic_source=local_source,
            exact_source=local_exact_source,
        )
        local_plans.append(
            FiniteElementPlan(
                local_mesh,
                source_plan.fields,
                coordinate_spec=local_spec,
                precision_policy=source_plan.precision_policy,
                coefficient_dtype=source_plan.coefficient_dtype,
            )
        )
    programs = authority.prepare_discretizations(
        local_plans,
        axis_name=axis_name,
        message_capacity=message_capacity,
        numeric_version=numeric_version,
    )
    workspace.retain_owner((local_plans, programs))
    return authority, programs


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


# Non-integral cell weights are quantized to this integer total for GRAPH.
_GRAPH_WEIGHT_TOTAL = float(1 << 40)


def _graph_owners(
    part: MeshPart, weights: np.ndarray, policy: MeshPartitionPolicy, /
) -> tuple[np.ndarray, str]:
    """Native multilevel ownership of the face-adjacency graph.

    Integral cell weights are used exactly; other weights are quantized to
    ``max(1, rint(w * 2**40 / total))``, which the provenance records.
    """
    integral = bool(np.all(weights == np.rint(weights)) and np.sum(weights) <= 2**53)
    vertex_weights = (
        weights.astype(np.int64)
        if integral
        else np.maximum(
            1, np.rint(weights * (_GRAPH_WEIGHT_TOTAL / np.sum(weights)))
        ).astype(np.int64)
    )
    adjacency = _carrier_adjacency(part)
    result = partition_graph(
        WeightedCSRGraph(
            np.asarray(adjacency.offsets),
            np.asarray(adjacency.neighbors),
            vertex_weights=vertex_weights,
        ),
        GraphPartitionPlan(policy.part_count, maximum_imbalance=policy.maximum_imbalance),
    )
    evidence = result.evidence
    weighting = "integer-weights" if integral else "quantized-weights"
    return (
        np.asarray(result.parts),
        f"graph-multilevel:{evidence.status}:{weighting}:{evidence.backend}",
    )


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
            return _graph_owners(part, weights, policy)
        case MeshPartitionKind.METIS:
            if policy.part_count > weights.size:
                raise ValueError("part_count exceeds the number of cells.")
            adjacency = _carrier_adjacency(part)
            owner = metis_partition(
                np.asarray(adjacency.offsets),
                np.asarray(adjacency.neighbors),
                weights,
                policy.part_count,
                imbalance=policy.maximum_imbalance,
                seed=policy.metis_seed,
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
    GRAPH runs the native partitioner and needs the meshcore library; only the
    METIS comparison route raises :class:`MetisUnavailableError`.
    An already owner-local accepted carrier consumes its actual provider
    ownership; optional vectors then follow the canonical global cell-ID prefix.
    Repartitioning that carrier requires a validated migration/publication,
    never relabeling its bound storage as a different partition algorithm.
    """
    if not isinstance(part, MeshPart):
        raise TypeError("part must be MeshPart.")
    if not isinstance(policy, MeshPartitionPolicy):
        raise TypeError("policy must be MeshPartitionPolicy.")
    carrier = part.carrier
    if isinstance(carrier, CellMeshingResult) and carrier.mesh.storage is not None:
        mesh, storage = carrier.mesh, carrier.mesh.storage
        evidence, binding = carrier.collective_evidence, carrier.storage_binding
        if storage is None or evidence is None or binding is None:
            raise ValueError(
                "Owner-local source distribution requires its actual accepted storage theorem binding."
            )
        binding.require_storage(mesh, evidence)
        evidence.require_passed()
        if (
            policy.kind is not MeshPartitionKind.PROVIDER
            or policy.part_count != storage.partition_count
            or policy.halo_width > storage.neighborhood_depth
        ):
            raise ValueError(
                "Bound source ownership requires the provider route and its certified owner count and neighborhood."
            )
        count = storage.global_entity_counts[-1]
        global_ids = evidence.entity_ids[-1][:count]
        global_owners = evidence.entity_owners[-1][:count]
        if ownership is not None:
            requested = jnp.asarray(ownership)
            if requested.shape != global_owners.shape or not jnp.issubdtype(
                requested.dtype, jnp.integer
            ):
                raise ValueError(
                    "Source ownership must follow the exact canonical global cell-ID prefix."
                )
            if not bool(jax.device_get(jnp.all(requested == global_owners))):
                raise ValueError(
                    "Changing accepted source owners requires actual validated migration and publication."
                )
        summary = _owner_local_partition_summary(mesh)
        local_weights = None
        if cell_weights is not None:
            from ._scope import _local_logical_lookup

            weights = jnp.asarray(cell_weights, dtype=jnp.float64)
            if weights.shape != global_ids.shape or not bool(
                jax.device_get(jnp.all(jnp.isfinite(weights) & (weights > 0)))
            ):
                raise ValueError(
                    "Source load weights require finite positive values on the exact global cell-ID prefix."
                )
            local_weights, valid = _local_logical_lookup(
                global_ids, weights, storage.entity_global_ids[-1]
            )
            if not np.all(valid):
                raise ValueError(
                    "Source load weights lack an actual resident scientific cell."
                )
            summary = (
                jnp.bincount(
                    global_owners, weights=weights, length=storage.partition_count
                ),
                summary[1],
                summary[2],
            )
        return MeshDistribution(
            part,
            CellPartition(
                storage.entity_owner[-1], storage.partition_count, storage=storage
            ),
            halo_width=policy.halo_width,
            cell_weights=local_weights,
            partition_kind=MeshPartitionKind.PROVIDER,
            partition_provenance="accepted-source-ownership",
            _collective_summary=summary,
        )
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


def _restart_transport_binding(
    source: MeshDistribution,
    target: MeshDistribution,
    repack: SimplexRestartRepack,
    proof: SimplexRestartRepackProof,
    /,
) -> str:
    """Consume the all-owner mathematical repack proof and exact published inputs locally."""

    if not isinstance(repack, SimplexRestartRepack) or not isinstance(
        proof, SimplexRestartRepackProof
    ):
        raise TypeError(
            "Changed placement requires its actual numerical repack and prepared proof."
        )
    proof.require_receipt(repack)
    original = repack.source_result
    before, after = source.part.carrier, target.part.carrier
    if (
        original is None
        or not isinstance(before, CellMeshingResult)
        or not isinstance(after, CellMeshingResult)
    ):
        raise ValueError(
            "Restart transport must bind actual accepted source and target results."
        )
    evidence = after.collective_evidence
    if not isinstance(evidence, CollectiveMeshEvidence) or (
        original.result_id != before.result_id
        or original.mesh.mesh_id != before.mesh.mesh_id
        or evidence.source.result_id != before.result_id
        or evidence.source_mesh_id != before.mesh.mesh_id
        or evidence.partition_count != target.partition.part_count
        or repack.parts.part_count != target.partition.part_count
        or evidence.layout.signature_id != repack.layout.signature_id
        or repack.cell_migration_counts is None
        or repack.cell_migration_counts.shape
        != (
            source.partition.part_count,
            target.partition.part_count,
        )
    ):
        raise ValueError(
            "Restart transport lost its accepted source, target or rectangular ownership binding."
        )
    initial = jax.tree_util.tree_leaves(evidence.initial_states)
    actual = jax.tree_util.tree_leaves(repack.states)
    if len(initial) != len(actual) or any(
        left is not right for left, right in zip(initial, actual, strict=True)
    ):
        raise ValueError(
            "Restart target does not consume the exact prepared numerical initial forest."
        )
    banks = dict(evidence.logical_arrays)
    for name, value in (
        ("placement/requested_target_owners", repack.requested_target_owners),
        ("placement/initial_solver_cell_owners", repack.solver_cell_owners),
        ("placement/cell_saved_locations", repack.cell_saved_locations),
        ("placement/vertex_saved_locations", repack.vertex_saved_locations),
        ("placement/cell_migration_counts", repack.cell_migration_counts),
    ):
        if name not in banks or banks[name] is not value:
            raise ValueError(
                "Restart target lost an exact independently validated placement receipt bank."
            )
    return proof.content_id


class MeshDistributionTransition(StrictModule, NonTrainableState):
    """Ownership carried through mesh lineage onto an adapted part revision.

    Each lineage route is one message slot; slots are ordered by (source rank,
    target rank, target global ID, source global ID). ``send`` gathers source
    cells into slots and ``receive`` reduces slots onto target cells, so
    :meth:`transfer` moves cell data exactly as rank-to-rank migration would.
    ``message_offsets`` groups slots by the source-rank-major pair
    ``source_rank * target.part_count + target_rank``. Its rectangular traffic
    table has one row per old owner and one column per new owner. Global IDs are
    never renumbered; changed owner counts consume an actual numerical repack proof.
    """

    source: MeshDistribution
    target: MeshDistribution
    send: EdgeRelation
    source_cell_global_ids: Array
    source_cell_count: int = eqx.field(static=True)
    receive: EdgeRelation
    message_source_ranks: Array
    message_target_ranks: Array
    message_offsets: Array
    migration_counts: Array
    target_defined: Array
    migrated_cells: Array
    migration_volume: Array
    rebalanced: bool = eqx.field(static=True)
    local_scope: bool = eqx.field(static=True)
    lineage_id: str = eqx.field(static=True)
    restart_repack_id: str | None = eqx.field(static=True)
    restart_source_result_id: str | None = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)

    @checked
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
        _source_cell_owners: ArrayLike | None = None,
        _source_cell_ids: ArrayLike | None = None,
        restart_repack: SimplexRestartRepack | None = None,
        restart_proof: SimplexRestartRepackProof | None = None,
    ) -> None:
        if not isinstance(source, MeshDistribution) or not isinstance(
            target, MeshDistribution
        ):
            raise TypeError("Transition endpoints must be MeshDistribution values.")
        parts = source.partition.part_count
        target_parts = target.partition.part_count
        repack_id = None
        if restart_repack is None:
            if restart_proof is not None or target_parts != parts:
                raise ValueError(
                    "Changed owner counts require an actual independently validated restart repack."
                )
        else:
            if restart_proof is None:
                raise ValueError(
                    "Restart transport requires its prepared all-owner numerical proof."
                )
            repack_id = _restart_transport_binding(
                source, target, restart_repack, restart_proof
            )
        sources = np.asarray(source_rows, dtype=np.int64)
        targets = np.asarray(target_rows, dtype=np.int64)
        source_count = (
            source.global_cell_count
            if source.local_scope
            else source.cell_global_ids.size
        )
        target_count = target.cell_global_ids.size
        if (
            sources.ndim != 1
            or targets.shape != sources.shape
            or np.any((sources < 0) | (sources >= source_count))
            or np.any((targets < 0) | (targets >= target_count))
        ):
            raise ValueError("Transition routes must address native source/target rows.")
        if source.local_scope:
            carrier = source.part.carrier
            if not isinstance(carrier, CellMeshingResult) or carrier.mesh.storage is None:
                raise ValueError(
                    "Logical source ownership requires its accepted mesh storage."
                )
            evidence = carrier.collective_evidence
            if (
                evidence is None
                or source_count != evidence.global_entity_counts[-1]
                or source.partition.global_cell_count != source_count
                or source.partition.storage_id != carrier.mesh.storage.storage_id
            ):
                raise ValueError(
                    "Logical source ownership lost its consumed global entity count."
                )
            evidence.require_passed()
            native_partition_id = canonical_fingerprint(
                {
                    "kind": "owner-local-cell-partition",
                    "cell_owner": array_tree_fingerprint(source.partition.cell_owner),
                    "part_count": parts,
                    "storage": source.partition.storage_id,
                    "global_cell_count": source_count,
                }
            )
            if (
                native_partition_id != source.partition.partition_id
                or canonical_fingerprint(array_tree_fingerprint(source.cell_global_ids))
                != source.cell_content_id
            ):
                raise ValueError(
                    "The source local view lost its owning native cache binding."
                )
            source_ids = (
                evidence.entity_ids[-1]
                if _source_cell_ids is None
                else jnp.asarray(_source_cell_ids)
            )
            source_owner = (
                evidence.entity_owners[-1]
                if _source_cell_owners is None
                else jnp.asarray(_source_cell_owners)
            )
            if (
                source_ids.shape != evidence.entity_ids[-1].shape
                or source_owner.shape != source_ids.shape
                or source_ids.dtype != jnp.int64
                or source_owner.dtype != jnp.int32
            ):
                raise ValueError(
                    "Source logical caches require the actual canonical ID/owner bank."
                )
            shapes = {"cell_ids": (source_count,), "cell_owner": (source_count,)}
            actual = logical_array_value_collection_digest(
                {"cell_ids": source_ids, "cell_owner": source_owner},
                logical_shapes=shapes,
            )
            expected = logical_array_value_collection_digest(
                {
                    "cell_ids": evidence.entity_ids[-1],
                    "cell_owner": evidence.entity_owners[-1],
                },
                logical_shapes=shapes,
            )
            if actual != expected:
                raise ValueError(
                    "Source logical ownership/ID receipt differs from its accepted source."
                )
            # Only transport occurrences are lowered. The scientific source
            # bank stays globally sharded, irrespective of local block order.
            sender = np.asarray(
                jax.device_get(source_owner[jnp.asarray(sources)]), dtype=np.int32
            )
            routed_source_ids = np.asarray(
                jax.device_get(source_ids[jnp.asarray(sources)]), dtype=np.int64
            )
            record = lineage.entity_lineage(carrier.mesh.topological_dimension)
            if not np.array_equal(
                routed_source_ids, np.asarray(record.source_global_ids)
            ) or not np.array_equal(
                np.asarray(target.cell_global_ids)[targets],
                np.asarray(record.target_global_ids),
            ):
                raise ValueError(
                    "Logical transport occurrences do not bind the consumed cell lineage."
                )
        else:
            source_owner = (
                np.asarray(source.partition.cell_owner, dtype=np.int32)
                if _source_cell_owners is None
                else np.asarray(_source_cell_owners, dtype=np.int32)
            )
            source_ids = (
                np.asarray(source.cell_global_ids, dtype=np.int64)
                if _source_cell_ids is None
                else np.asarray(_source_cell_ids, dtype=np.int64)
            )
            source_partition_payload = {
                "kind": "cell-partition",
                "cell_owner": array_tree_fingerprint(source_owner),
                "part_count": parts,
            }
            if (
                canonical_fingerprint(source_partition_payload)
                != source.partition.partition_id
                or canonical_fingerprint(array_tree_fingerprint(source_ids))
                != source.cell_content_id
            ):
                raise ValueError(
                    "Prepared source ownership/ID cache differs from its bound source."
                )
            sender = source_owner[sources]
            routed_source_ids = source_ids[sources]
        receiver = np.asarray(target.partition.cell_owner, dtype=np.int64)[targets]
        if np.any((sender < 0) | (sender >= parts)) or np.any(
            (receiver < 0) | (receiver >= target_parts)
        ):
            raise ValueError(
                "Transition sender and receiver ranks must belong to their exact respective owner groups."
            )
        order = np.lexsort(
            (
                routed_source_ids,
                np.asarray(target.cell_global_ids)[targets],
                receiver,
                sender,
            )
        )
        sources, targets = sources[order], targets[order]
        sender, receiver = sender[order], receiver[order]
        slots = np.arange(sources.size, dtype=np.int32)
        pair_counts = np.bincount(
            sender * target_parts + receiver, minlength=parts * target_parts
        )
        offsets = np.concatenate(([0], np.cumsum(pair_counts)))
        remote = np.unique(targets[sender != receiver])
        volume = np.sum(np.asarray(target.cell_weights)[remote])
        self.source = source
        self.target = target
        self.source_cell_global_ids = jnp.asarray(source_ids, dtype=jnp.int64)
        self.source_cell_count = source_count
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
            pair_counts.reshape(parts, target_parts), dtype=jnp.int64
        )
        self.local_scope = source.local_scope or target.local_scope
        self.target_defined = jnp.asarray(
            np.bincount(targets, minlength=target_count) > 0
        )
        self.migrated_cells = jnp.asarray(remote.size, dtype=jnp.int64)
        self.migration_volume = jnp.asarray(volume, dtype=jnp.float64)
        self.rebalanced = bool(rebalanced)
        self.lineage_id = lineage.lineage_id
        self.restart_repack_id = repack_id
        self.restart_source_result_id = (
            None if restart_proof is None else restart_proof.source_result_id
        )
        self.transition_id = canonical_fingerprint(
            {
                "kind": "mesh-distribution-transition",
                "source": source.distribution_id,
                "target": target.distribution_id,
                "lineage": lineage.lineage_id,
                "sources": array_tree_fingerprint(sources),
                "targets": array_tree_fingerprint(targets),
                "rebalanced": self.rebalanced,
                **({} if repack_id is None else {"restart_repack": repack_id}),
            }
        )

    def _rank(self, rank: int, parts: int, /) -> int:
        index = operator.index(rank)
        if not 0 <= index < parts:
            raise ValueError("Transition rank is out of range.")
        return index

    def sent_by(self, rank: int, /) -> Array:
        """Message slots sent by ``rank``, grouped by receiving rank."""
        index = self._rank(rank, self.source.partition.part_count)
        parts = self.target.partition.part_count
        offsets = np.asarray(self.message_offsets)
        return jnp.arange(offsets[index * parts], offsets[(index + 1) * parts])

    def received_by(self, rank: int, /) -> Array:
        """Message slots received by ``rank``, grouped by sending rank."""
        index = self._rank(rank, self.target.partition.part_count)
        parts = self.target.partition.part_count
        offsets = np.asarray(self.message_offsets)
        pairs = np.arange(self.source.partition.part_count) * parts + index
        starts, stops = offsets[pairs], offsets[pairs + 1]
        counts = stops - starts
        shift = np.repeat(starts - (np.cumsum(counts) - counts), counts)
        return jnp.asarray(np.arange(np.sum(counts)) + shift, dtype=jnp.int32)

    def transfer(
        self, source_values: ArrayLike, /, *, reduction: RouteReduction = "mean"
    ) -> Array:
        """Migrate source-bank cell data onto native target cells.

        Owner-local predecessors use the canonical global ID prefix exposed by
        ``source_cell_global_ids``; compact predecessors retain native order.

        Preserved and refined cells copy their parent value; merged cells reduce
        their sources with ``reduction``. Created cells (``~target_defined``)
        receive zero and must be initialized by the consumer.
        """
        values = jnp.asarray(source_values)
        if values.ndim == 0 or values.shape[0] != self.source_cell_count:
            raise ValueError(
                "Transferred values must follow the bound source cell bank prefix."
            )
        return route_reduce(
            self.receive, gather_routes(self.send, values), reduction=reduction
        )

    def composition_transport(
        self, source: CompositionEntry, target: CompositionEntry, /
    ) -> CompositionTransport:
        """Ownership-migration evidence of this transition for one cell-state entry.

        Entry values are native-order rows of extensive cell content (leading
        axis = cells); `source` lives on the source distribution and `target` on
        the target distribution (entry structure identities are their
        `distribution_id`). A migration moves every physical row between owners
        and never creates or duplicates content: created target rows need an
        explicit physical remap or initialization rule, and deleted or refined
        source rows (a parent copied into several children) need a physical
        remap, so all three are refused. Merged rows sum their sources, the only
        route reduction that conserves extensive content. Partition cell weights
        are load-balance costs, not measures, so they never scale content. The
        transport reports per-component content over all rows with the
        recursive-summation roundoff bound, and succeeds only when `target` is
        the migration image of `source` within roundoff, so a staged value
        cannot borrow this route's evidence.
        """

        # Lazy: the lifecycle package sits above the meshing owners.
        from ..lifecycle import CompositionEntry, CompositionTransport

        if not isinstance(source, CompositionEntry) or not isinstance(
            target, CompositionEntry
        ):
            raise TypeError("Composition transports bind CompositionEntry values.")
        if (
            source.structure_id != self.source.distribution_id
            or target.structure_id != self.target.distribution_id
        ):
            raise ValueError(
                "Composition entries do not live on this transition's distributions."
            )
        if not np.all(np.asarray(self.target_defined)):
            raise ValueError(
                "Created target rows hold no migrated content; they need an explicit "
                "physical remap or initialization rule, never an ownership migration."
            )
        sent = np.bincount(
            np.asarray(self.send.source_indices), minlength=self.send.source_size
        )
        if np.any(sent != 1):
            raise ValueError(
                "Ownership migration moves every source row exactly once; deleted or "
                "refined rows need an explicit physical remap."
            )
        values = jnp.asarray(source.value)
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            raise TypeError("Migrated cell content must be a real or complex array.")
        image = self.transfer(values, reduction="sum")
        staged = jnp.asarray(target.value)
        if staged.shape != image.shape or staged.dtype != image.dtype:
            raise ValueError("Staged target does not match the migrated row layout.")
        eps = jnp.finfo(jnp.real(image).dtype).eps
        exact = jnp.abs(staged - image) <= 100 * eps * jnp.max(jnp.abs(image))
        finite = jnp.all(jnp.isfinite(image)) & jnp.all(jnp.isfinite(staged))
        # Both sums carry at most (rows - 1) eps sum|v| recursive-summation error.
        tolerance = 2 * values.shape[0] * eps * jnp.sum(jnp.abs(values), axis=0)
        return CompositionTransport(
            "ownership-migration",
            (source.entry_id,),
            (target,),
            source_structure_ids=(self.source.distribution_id,),
            route_id=self.transition_id,
            successful=finite & jnp.all(exact),
            source_content=jnp.sum(values, axis=0),
            target_content=jnp.sum(staged, axis=0),
            content_tolerance=tolerance,
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


def _lexicographic_bound(table: Array, queries: Array, upper: bool, /) -> Array:
    """Vector lower/upper bounds without a query-by-table allocation."""
    size = table.shape[0]
    low = jnp.zeros((queries.shape[0],), dtype=jnp.int32)
    high = jnp.full_like(low, size)

    def proceed(bounds: tuple[Array, Array]) -> Array:
        return jnp.any(bounds[0] < bounds[1])

    def step(bounds: tuple[Array, Array]) -> tuple[Array, Array]:
        left, right = bounds
        midpoint = (left + right) // 2
        row = table[jnp.minimum(midpoint, size - 1)]
        smaller = jnp.zeros_like(left, dtype=jnp.bool_)
        equal = jnp.ones_like(smaller)
        for column in range(table.shape[1]):
            smaller |= equal & (row[:, column] < queries[:, column])
            equal &= row[:, column] == queries[:, column]
        move = smaller | (equal & upper)
        active = left < right
        return (
            jnp.where(active & move, midpoint + 1, left),
            jnp.where(active & ~move, midpoint, right),
        )

    return jax.lax.while_loop(proceed, step, (low, high))[0]


def _facet_packets(state: AdaptiveSimplexState, /) -> tuple[Array, Array]:
    width = state.mesh.cells.shape[1]
    columns = jnp.asarray(_facet_columns(width), dtype=jnp.int32)
    identifiers = state.mesh.vertex_ids[state.mesh.cells[:, columns]]
    signs = jnp.ones(identifiers.shape[:2], dtype=jnp.int32)
    for left in range(width - 1):
        for right in range(left + 1, width - 1):
            signs *= jnp.where(identifiers[..., left] > identifiers[..., right], -1, 1)
    signs *= jnp.where(jnp.arange(width) % 2 == 0, 1, -1)[None, :]
    active = jnp.repeat(state.mesh.cell_active, width)
    keys = jnp.sort(identifiers, axis=2).reshape((-1, width - 1))
    keys = jnp.where(active[:, None], keys, jnp.iinfo(jnp.int64).max)
    return keys, jnp.where(active, signs.reshape(-1), 0)


def _facet_matches(
    keys: Array, packet: Array, orientations: Array, /
) -> tuple[Array, Array]:
    columns = tuple(packet[:, column] for column in range(packet.shape[1]))
    sorted_values = jax.lax.sort((*columns, orientations), num_keys=packet.shape[1])
    table = jnp.stack(sorted_values[:-1], axis=1)
    first = _lexicographic_bound(table, keys, False)
    last = _lexicographic_bound(table, keys, True)
    prefix = jnp.concatenate(
        (jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(sorted_values[-1], dtype=jnp.int32))
    )
    valid = keys[:, 0] != jnp.iinfo(jnp.int64).max
    return (
        jnp.where(valid, last - first, 0),
        jnp.where(valid, prefix[last] - prefix[first], 0),
    )


def _proof_packet_permutation(
    source: int, target: int, count: int, logical_batch: bool, /
) -> tuple[tuple[int, int], ...]:
    if not logical_batch:
        return ((source, target),)
    # Named vmap axes require a complete permutation. The swapped reverse
    # packet and identity lanes are ignored by the same recipient predicate.
    return tuple(
        (rank, target if rank == source else source if rank == target else rank)
        for rank in range(count)
    )


def _collective_facets(
    state: AdaptiveSimplexState,
    neighbor_pairs: tuple[tuple[int, int], ...],
    axis: str,
    part_count: int,
    logical_batch: bool,
    /,
) -> tuple[Array, Array]:
    keys, orientations = _facet_packets(state)
    counts, signs = _facet_matches(keys, keys, orientations)
    rank = jax.lax.axis_index(axis)
    for source, target in neighbor_pairs:
        permutation = _proof_packet_permutation(source, target, part_count, logical_batch)
        received = jax.lax.ppermute(keys, axis, permutation)
        received_signs = jax.lax.ppermute(orientations, axis, permutation)
        other_counts, other_signs = _facet_matches(keys, received, received_signs)
        counts += jnp.where(rank == target, other_counts, 0)
        signs += jnp.where(rank == target, other_signs, 0)
    return counts, signs


def _subdivision_checks(
    state: AdaptiveSimplexState, initial: AdaptiveSimplexState, exterior: Array, /
) -> tuple[Array, Array, Array, Array]:
    work = _work(state)
    cells = work.cell_ids.shape[0]
    vertices = work.vertex_ids.shape[0]
    width = state.mesh.cells.shape[1]
    lanes = jnp.arange(cells, dtype=jnp.int32)
    vertex_lanes = jnp.arange(vertices, dtype=jnp.int32)
    original_cells = lanes < initial.cursors[1]
    original_vertices = vertex_lanes < initial.cursors[0]
    allocated = lanes < work.cursors[1]
    allocated_vertices = vertex_lanes < work.cursors[0]
    living = allocated & ~work.retired
    first = jnp.clip(work.children[:, 0], 0, cells - 1)
    second = jnp.clip(work.children[:, 1], 0, cells - 1)
    has_children = (work.children[:, 0] >= 0) & (work.children[:, 1] >= 0)
    restored_history = (
        living
        & work.cell_active
        & has_children
        & work.retired[first]
        & work.retired[second]
    )
    split = living & has_children & ~work.cell_active
    family = split | restored_history
    parent = jnp.clip(work.parents // 2, 0, cells - 1)
    expected_first, expected_second, tags = _children(
        work, jnp.clip(work.bisection_vertices, 0, vertices - 1), width - 1
    )
    roots = jnp.all(
        ~original_cells
        | (
            (work.cell_ids == initial.mesh.cell_ids)
            & jnp.all(work.cells == initial.mesh.cells, axis=1)
            & jnp.all(work.tuples == initial.tuples, axis=1)
            & (work.tags == initial.tags)
            & (work.blocks == initial.blocks)
            & (work.cell_classes == initial.cell_classes)
            & jnp.all(work.facet_classes == initial.facet_classes, axis=1)
            & (work.parents == initial.parents)
            & (work.generations == initial.generations)
            & (~initial.retired | work.retired)
            & (~initial.retired | ~work.cell_active)
        )
    ) & jnp.all(
        ~original_vertices
        | (
            (work.vertex_ids == initial.mesh.vertex_ids)
            & jnp.all(work.coordinates == initial.mesh.coordinates, axis=1)
            & jnp.all(work.vertex_parents == initial.vertex_parents, axis=1)
        )
    )
    descendants = (
        jnp.all(
            ~living
            | (
                (work.cell_active == ~split)
                & (~has_children | family)
                & ((work.parents < 0) == (original_cells & (initial.parents < 0)))
                & (
                    (work.parents < 0)
                    | (
                        (work.parents >= 0)
                        & (parent < lanes)
                        & (work.generations == work.generations[parent] + 1)
                        & (lanes == work.children[parent, work.parents % 2])
                    )
                )
            )
        )
        & jnp.all(
            ~family
            | (
                (work.children[:, 0] > lanes)
                & (work.children[:, 1] > lanes)
                & (work.children[:, 0] < work.cursors[1])
                & (work.children[:, 1] < work.cursors[1])
                & (work.parents[first] == 2 * lanes)
                & (work.parents[second] == 2 * lanes + 1)
                & jnp.where(
                    split,
                    ~work.retired[first] & ~work.retired[second],
                    work.retired[first] & work.retired[second],
                )
                & jnp.all(work.cells[first] == expected_first[1], axis=1)
                & jnp.all(work.cells[second] == expected_second[1], axis=1)
                & jnp.all(work.facet_classes[first] == expected_first[2], axis=1)
                & jnp.all(work.facet_classes[second] == expected_second[2], axis=1)
                & (work.tags[first] == tags)
                & (work.tags[second] == tags)
                & (work.cell_classes[first] == work.cell_classes)
                & (work.cell_classes[second] == work.cell_classes)
            )
        )
        & ~jnp.any(work.retired & work.cell_active)
        & ~jnp.any(allocated & (work.parents < 0) & work.retired)
        & (work.cursors[0] >= initial.cursors[0])
        & (work.cursors[1] >= initial.cursors[1])
    )
    endpoints = jnp.clip(work.vertex_parents, 0, vertices - 1)
    midpoint_coordinates = (
        0.5 * work.coordinates[endpoints[:, 0]] + 0.5 * work.coordinates[endpoints[:, 1]]
    )
    midpoints = jnp.all(
        ~allocated_vertices
        | (
            jnp.all(jnp.isfinite(work.coordinates), axis=1)
            & (
                original_vertices
                | (
                    jnp.all(work.vertex_parents >= 0, axis=1)
                    & jnp.all(work.vertex_parents < vertex_lanes[:, None], axis=1)
                    & jnp.all(work.coordinates == midpoint_coordinates, axis=1)
                )
            )
        )
    )
    low, high = _refinement_edges(work)
    midpoint = jnp.clip(work.bisection_vertices, 0, vertices - 1)
    midpoints &= jnp.all(
        ~family
        | jnp.all(
            jnp.sort(work.vertex_parents[midpoint], axis=1)
            == jnp.sort(jnp.stack((low, high), axis=1), axis=1),
            axis=1,
        )
    )
    initial_work = _work(initial)
    edge_codes = jnp.minimum(endpoints[:, 0], endpoints[:, 1]).astype(
        jnp.int64
    ) * vertices + jnp.maximum(endpoints[:, 0], endpoints[:, 1]).astype(jnp.int64)
    protected_split, _ = _members(initial_work.protected_codes, edge_codes)
    constraints = (
        jnp.array_equal(work.protected_codes, initial_work.protected_codes)
        & jnp.all(
            ~original_vertices | (work.vertex_protected == initial_work.vertex_protected)
        )
        & jnp.all(
            ~initial_work.vertex_protected
            | (
                work.vertex_active
                & jnp.all(work.coordinates == initial_work.coordinates, axis=1)
            )
        )
        & ~jnp.any(allocated_vertices & ~original_vertices & protected_split)
    )
    midpoints &= constraints
    boundary = jnp.where(
        initial.mesh.cell_active[:, None], exterior.reshape((cells, width)), 0
    )

    initial_depth = jnp.max(jnp.where(original_cells, initial.generations, 0))

    def restore(step: Array, flags: Array) -> Array:
        inherited = _restored_facet_classes(initial_work._replace(facet_classes=flags))
        selected = (
            original_cells
            & ~initial.mesh.cell_active
            & ~initial.retired
            & (initial.generations == initial_depth - step)
        )
        return jnp.where(selected[:, None], inherited, flags)

    boundary = jax.lax.fori_loop(0, initial_depth + 1, restore, boundary)

    def inherit(level: Array, flags: Array) -> Array:
        first_child, second_child, _ = _children(
            work._replace(facet_classes=flags),
            midpoint,
            width - 1,
        )
        selected = split & (work.generations == level)
        first_slots = jnp.where(selected, first, cells)
        second_slots = jnp.where(selected, second, cells)
        return (
            flags.at[first_slots]
            .set(first_child[2], mode="drop")
            .at[second_slots]
            .set(second_child[2], mode="drop")
        )

    boundary = jax.lax.fori_loop(
        0, jnp.max(jnp.where(allocated, work.generations, 0)) + 1, inherit, boundary
    )
    return roots, descendants, midpoints, boundary.reshape(-1).astype(jnp.bool_)


def _bisection_part_checks(
    state: AdaptiveSimplexState,
    initial: AdaptiveSimplexState,
    authoritative_exterior: Array,
    part_count: int,
    neighbor_pairs: tuple[tuple[int, int], ...],
    axis: str,
    logical_batch: bool = False,
    /,
) -> Array:
    """One saved logical owner's numerical checks on a named collective axis."""
    initial_counts, _ = _collective_facets(
        initial, neighbor_pairs, axis, part_count, logical_batch
    )
    authoritative_exterior = authoritative_exterior.reshape(-1)
    initial_facets_active = jnp.repeat(
        initial.mesh.cell_active, initial.mesh.cells.shape[1]
    )
    source_facets = jnp.all(
        ~initial_facets_active
        | (initial_counts == jnp.where(authoritative_exterior, 1, 2))
    )
    roots, descendants, midpoints, exterior = _subdivision_checks(
        state, initial, authoritative_exterior.astype(jnp.int32)
    )
    roots &= source_facets
    counts, signs = _collective_facets(
        state, neighbor_pairs, axis, part_count, logical_batch
    )
    active_facets = jnp.repeat(state.mesh.cell_active, state.mesh.cells.shape[1])
    reciprocal = jnp.all(
        ~active_facets
        | ((counts == jnp.where(exterior, 1, 2)) & (exterior | (signs == 0)))
    )
    ids = jnp.sort(
        jnp.where(state.mesh.cell_active, state.mesh.cell_ids, jnp.iinfo(jnp.int64).max)
    )
    valid = ids != jnp.iinfo(jnp.int64).max
    unique = ~jnp.any(valid[1:] & (ids[1:] == ids[:-1]))
    ring = tuple((rank, (rank + 1) % part_count) for rank in range(part_count))

    def circulate(_: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        packet, accepted = carry
        packet = jax.lax.ppermute(packet, axis, ring)
        position = jnp.minimum(jnp.searchsorted(packet, ids), packet.shape[0] - 1)
        return packet, accepted & ~jnp.any(valid & (packet[position] == ids))

    _, unique = jax.lax.fori_loop(0, part_count - 1, circulate, (ids, unique))
    vertex_lanes = jnp.arange(state.mesh.vertex_ids.shape[0])
    vertex_ids = jnp.where(
        vertex_lanes < state.cursors[0],
        state.mesh.vertex_ids,
        jnp.iinfo(jnp.int64).max,
    )
    shared = jnp.asarray(True)
    rank = jax.lax.axis_index(axis)
    for source, target in neighbor_pairs:
        permutation = _proof_packet_permutation(source, target, part_count, logical_batch)
        other_ids = jax.lax.ppermute(vertex_ids, axis, permutation)
        other_points = jax.lax.ppermute(state.mesh.coordinates, axis, permutation)
        position = jnp.minimum(
            jnp.searchsorted(other_ids, vertex_ids), other_ids.shape[0] - 1
        )
        match = (other_ids[position] == vertex_ids) & (vertex_lanes < state.cursors[0])
        shared &= (rank != target) | jnp.all(
            ~match | jnp.all(state.mesh.coordinates == other_points[position], axis=1)
        )
    status = _terminal(_work(state), None) == 0
    return jnp.stack((roots, descendants, midpoints, unique, reciprocal, shared, status))


def _collective_bisection_checks(
    parts: AdaptiveSimplexParts,
    states: AdaptiveSimplexState,
    initial_states: AdaptiveSimplexState,
    initial_exterior: Array,
    /,
) -> Array:
    spec = PartitionSpec(parts.axis_name)

    def local(
        state_block: AdaptiveSimplexState,
        initial_block: AdaptiveSimplexState,
        exterior_block: Array,
    ) -> Array:
        state = jax.tree_util.tree_map(lambda value: value[0], state_block)
        initial = jax.tree_util.tree_map(lambda value: value[0], initial_block)
        return _bisection_part_checks(
            state,
            initial,
            exterior_block[0],
            parts.part_count,
            parts.neighbor_pairs,
            parts.axis_name,
        )[None]

    return jax.shard_map(
        local,
        mesh=parts.mesh,
        in_specs=(spec, spec, spec),
        out_specs=spec,
        check_vma=False,
    )(states, initial_states, initial_exterior)


def replay_logical_bisection_checks(
    states: AdaptiveSimplexState,
    initial_states: AdaptiveSimplexState,
    initial_exterior: Array,
    /,
    *,
    neighbor_pairs: tuple[tuple[int, int], ...],
    axis_name: str,
) -> Array:
    """Replay saved logical owners on any available physical device placement.

    The mapped axis is the actual retained partition axis, not the current
    hardware mesh. All facet exchanges, identity circulation and shared-point
    comparisons are recomputed from those numerical records.
    """
    count = states.mesh.cell_ids.shape[0]
    if (
        states.mesh.cell_ids.ndim != 2
        or initial_states.mesh.cell_ids.shape != states.mesh.cell_ids.shape
        or initial_exterior.shape != initial_states.mesh.cells.shape
        or not isinstance(axis_name, str)
        or not axis_name
        or any(
            source == target or not 0 <= source < count or not 0 <= target < count
            for source, target in neighbor_pairs
        )
    ):
        raise ValueError(
            "Logical replay requires matching saved owner records and their real neighbor graph."
        )

    def local(
        state: AdaptiveSimplexState,
        initial: AdaptiveSimplexState,
        exterior: Array,
    ) -> Array:
        return _bisection_part_checks(
            state,
            initial,
            exterior,
            count,
            neighbor_pairs,
            axis_name,
            True,
        )

    return jax.vmap(local, axis_name=axis_name)(states, initial_states, initial_exterior)


_compiled_bisection_checks = eqx.filter_jit(_collective_bisection_checks)


class AffineBisectionSourceWitness(StrictModule, NonTrainableState):
    """Validated immutable source-map premise prepared before distributed execution."""

    source_result_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    source_layout_id: str = eqx.field(static=True)
    source_coordinate_geometry_id: str = eqx.field(static=True)
    scientific_source_result_id: str = eqx.field(static=True)
    scientific_certification_id: str = eqx.field(static=True)
    predecessor_evidence_id: str | None = eqx.field(static=True)
    witness_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: CellMeshingResult,
        /,
        *,
        uniform_refinement: BisectionUniformRefinement | None,
    ) -> None:
        from ..discretization._cell_geometry import coordinate_lagrange_element
        from ..discretization._cell_geometry_validity import cell_geometry_id
        from ._initial_certification import InitialCollectiveMeshEvidence

        scientific_source = require_original_meshing_source(
            source if uniform_refinement is None else uniform_refinement.source,
        )
        if isinstance(scientific_source, InitialCollectiveMeshEvidence):
            scientific_source.require_passed()
            from ..discretization._coordinate_enclosure import coordinate_source_signature

            element = coordinate_lagrange_element("triangle", 1)
            basis = np.frombuffer(
                bytes.fromhex(
                    canonical_fingerprint(
                        {
                            "signature": coordinate_source_signature(element),
                            "arrays": array_tree_fingerprint(element),
                        }
                    )
                ),
                dtype=np.uint8,
            )
            source_banks = dict(scientific_source.logical_arrays)
            count = scientific_source.global_entity_counts[-1]
            if not bool(
                jax.device_get(
                    jnp.all(
                        source_banks["geometry/source_basis/surface"][:count]
                        == jnp.asarray(basis)
                    )
                )
            ):
                raise ValueError(
                    "Initial scientific source lost its actual affine basis definition."
                )
            scientific_id = scientific_source.evidence_id
            premise_id = scientific_source.evidence_id
        else:
            certification = scientific_source.certification
            if certification is None or certification.embedding is None:
                raise ValueError(
                    "The original scientific source embedding certification is absent."
                )
            from .._meshcore import current_native_host_workspace
            from ._collective_geometry import _original_p1_vertex_images

            workspace = current_native_host_workspace()
            if workspace is None:
                raise RuntimeError(
                    "Scientific P1 source validation requires its actual native host owner."
                )
            workspace.retain_owner(scientific_source)
            retained = workspace.bound
            try:
                _original_p1_vertex_images(scientific_source)
            finally:
                workspace.set_bound(retained)
            scientific_id = scientific_source.result_id
            premise_id = certification.report_id
        source.geometry.resolve(source.mesh)
        coordinate_geometry_id = cell_geometry_id(source.geometry)
        if source.collective_evidence is not None and (
            coordinate_geometry_id != source.collective_evidence.coordinate_geometry_id
        ):
            raise ValueError(
                "The accepted predecessor lost its actual original coefficient restriction map."
            )
        self.source_result_id = source.result_id
        self.source_geometry_id = source.mesh.geometry_id
        self.source_layout_id = source.geometry.geometry_layout_id
        self.source_coordinate_geometry_id = coordinate_geometry_id
        self.scientific_source_result_id = scientific_id
        self.scientific_certification_id = premise_id
        self.predecessor_evidence_id = (
            None
            if source.collective_evidence is None
            else source.collective_evidence.evidence_id
        )
        self.witness_id = canonical_fingerprint(
            {
                "kind": "prepared-affine-bisection-source",
                "source": source.result_id,
                "geometry": source.mesh.geometry_id,
                "geometry_layout": source.geometry.geometry_layout_id,
                "coordinate_geometry": self.source_coordinate_geometry_id,
                "scientific_source": self.scientific_source_result_id,
                "certification": self.scientific_certification_id,
                "predecessor_evidence": self.predecessor_evidence_id,
                "uniform_lineage": None
                if uniform_refinement is None
                else uniform_refinement.lineage_id,
            }
        )


class SimplexNeighborhoodWorkset(StrictModule, NonTrainableState):
    """Bounded owner/ghost packets with actual affine field ancestry."""

    cell_ids: Array
    cell_vertices: Array
    cell_coordinates: Array
    cell_owner: Array
    cell_classes: Array
    cell_exterior: Array
    cell_valid: Array
    root_cell_ids: Array
    source_cell_ids: Array
    source_vertex_ids: Array
    source_weights: Array
    status: Array


def _simplex_neighborhood_seed(
    state: AdaptiveSimplexState,
    initial: AdaptiveSimplexState,
    initial_exterior: Array,
    rank: Array,
    capacity: int,
    /,
) -> SimplexNeighborhoodWorkset:
    """Lower accepted midpoint DAGs into compact per-cell affine witnesses."""
    cells, width = state.mesh.cells.shape
    vertices = state.mesh.vertex_ids.shape[0]
    roots = jnp.full((cells,), -1, dtype=jnp.int32)

    def ancestor(slot: Array, values: Array) -> Array:
        parent = state.parents[slot] // 2
        root = jnp.where(
            initial.mesh.cell_active[slot],
            slot,
            jnp.where(parent < 0, -1, values[jnp.maximum(parent, 0)]),
        )
        return values.at[slot].set(root)

    roots = jax.lax.fori_loop(0, cells, ancestor, roots)
    active_ancestors = jnp.full((cells,), -1, dtype=jnp.int32)

    def target_ancestor(slot: Array, values: Array) -> Array:
        parent = state.parents[slot] // 2
        nearest = jnp.where(
            state.mesh.cell_active[slot],
            slot,
            jnp.where(parent < 0, -1, values[jnp.maximum(parent, 0)]),
        )
        return values.at[slot].set(nearest)

    active_ancestors = jax.lax.fori_loop(0, cells, target_ancestor, active_ancestors)
    lanes = jnp.arange(cells, dtype=jnp.int32)
    predecessor_support = (
        (
            (roots[:, None] == lanes[None, :])
            | ((roots[:, None] < 0) & (lanes[:, None] == active_ancestors[None, :]))
        )
        & initial.mesh.cell_active[None, :]
        & state.mesh.cell_active[:, None]
    )
    source_cells = jnp.where(predecessor_support, initial.mesh.cell_ids[None, :], -1)
    weights = jnp.zeros((cells, width, width), dtype=jnp.float64)
    source_ids = jnp.full((cells, width, width), -1, dtype=jnp.int64)
    vertex_lanes = jnp.arange(vertices, dtype=jnp.int32)

    def active_root_support(
        root: Array, carry: tuple[Array, Array]
    ) -> tuple[Array, Array]:
        support_ids, support_weights = carry
        root_ids = initial.mesh.vertex_ids[initial.mesh.cells[root]]
        basis = (
            (initial.mesh.vertex_ids[:, None] == root_ids[None, :])
            & (vertex_lanes[:, None] < initial.cursors[0])
        ).astype(jnp.float64)

        def midpoint(slot: Array, values: Array) -> Array:
            parents = jnp.clip(state.vertex_parents[slot], 0, vertices - 1)
            return values.at[slot].set(
                0.5 * values[parents[0]] + 0.5 * values[parents[1]]
            )

        basis = jax.lax.fori_loop(
            initial.cursors[0].astype(jnp.int32),
            state.cursors[0].astype(jnp.int32),
            midpoint,
            basis,
        )
        coefficients = basis[state.mesh.cells]
        keys = jnp.where(
            coefficients != 0, root_ids[None, None, :], jnp.iinfo(jnp.int64).max
        )
        order = jnp.argsort(keys, axis=2, stable=True)
        keys = jnp.take_along_axis(keys, order, axis=2)
        coefficients = jnp.take_along_axis(coefficients, order, axis=2)
        selected = (
            (roots == root) & state.mesh.cell_active & initial.mesh.cell_active[root]
        )
        return (
            jnp.where(
                selected[:, None, None],
                jnp.where(coefficients != 0, keys, -1),
                support_ids,
            ),
            jnp.where(selected[:, None, None], coefficients, support_weights),
        )

    def root_support(root: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        return jax.lax.cond(
            initial.mesh.cell_active[root]
            & jnp.any((roots == root) & state.mesh.cell_active),
            lambda values: active_root_support(root, values),
            lambda values: values,
            carry,
        )

    source_ids, weights = jax.lax.fori_loop(
        0, initial.cursors[1].astype(jnp.int32), root_support, (source_ids, weights)
    )
    coarsened = state.mesh.cell_active & (roots < 0)
    identity_ids = jnp.full((cells, width, width), -1, dtype=jnp.int64)
    identity_ids = identity_ids.at[:, :, 0].set(state.mesh.vertex_ids[state.mesh.cells])
    identity_weights = jnp.zeros_like(weights).at[:, :, 0].set(1.0)
    source_ids = jnp.where(coarsened[:, None, None], identity_ids, source_ids)
    weights = jnp.where(coarsened[:, None, None], identity_weights, weights)
    supported_vertices = (
        jnp.zeros((vertices,), dtype=jnp.bool_)
        .at[initial.mesh.cells.reshape(-1)]
        .max(jnp.repeat(initial.mesh.cell_active, width))
    )
    ancestry_valid = jnp.all(
        ~state.mesh.cell_active
        | (
            jnp.any(predecessor_support, axis=1)
            & jnp.all(jnp.sum(weights, axis=2) == 1.0, axis=1)
            & (~coarsened | jnp.all(supported_vertices[state.mesh.cells], axis=1))
        )
    )
    order = jnp.argsort(
        jnp.where(state.mesh.cell_active, state.mesh.cell_ids, jnp.iinfo(jnp.int64).max),
        stable=True,
    )
    active_count = jnp.sum(state.mesh.cell_active, dtype=jnp.int32)
    indices = order[:capacity]
    padding = max(0, capacity - cells)

    def gather(value: Array) -> Array:
        result = value[indices]
        return jnp.pad(result, ((0, padding), *((0, 0) for _ in result.shape[1:])))

    valid = jnp.arange(capacity) < active_count
    root_ids = jnp.where(roots >= 0, initial.mesh.cell_ids[jnp.maximum(roots, 0)], -1)
    ids = gather(state.mesh.cell_ids)
    _, _, _, exterior = _subdivision_checks(
        state, initial, initial_exterior.reshape(-1).astype(jnp.int32)
    )
    return SimplexNeighborhoodWorkset(
        jnp.where(valid, ids, jnp.iinfo(jnp.int64).max),
        gather(state.mesh.vertex_ids[state.mesh.cells]),
        gather(state.mesh.coordinates[state.mesh.cells]),
        jnp.full((capacity,), rank, dtype=jnp.int32),
        gather(state.cell_classes),
        gather(exterior.reshape((cells, width))),
        valid,
        gather(root_ids),
        jnp.where(valid[:, None], gather(source_cells), -1),
        gather(source_ids),
        gather(weights),
        jnp.where(
            active_count > capacity,
            int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED),
            0,
        ).astype(jnp.int32)
        | jnp.where(
            ancestry_valid, 0, int(AdaptiveSimplexStatus.INVALID_GEOMETRY)
        ).astype(jnp.int32),
    )


def _neighborhood_facet_keys(work: SimplexNeighborhoodWorkset, /) -> Array:
    width = work.cell_vertices.shape[1]
    columns = jnp.asarray(_facet_columns(width), dtype=jnp.int32)
    keys = jnp.sort(work.cell_vertices[:, columns], axis=2).reshape((-1, width - 1))
    return jnp.where(
        jnp.repeat(work.cell_valid, width)[:, None], keys, jnp.iinfo(jnp.int64).max
    )


def _merge_neighborhood_packets(
    resident: SimplexNeighborhoodWorkset,
    packet: SimplexNeighborhoodWorkset,
    frontier_keys: Array,
    /,
) -> SimplexNeighborhoodWorkset:
    capacity, width = resident.cell_vertices.shape
    packet_keys = _neighborhood_facet_keys(packet)
    matches, _ = _facet_matches(
        packet_keys, frontier_keys, jnp.zeros((frontier_keys.shape[0],), dtype=jnp.int32)
    )
    adjacent = jnp.any(matches.reshape((capacity, width)) > 0, axis=1)
    position = jnp.minimum(
        jnp.searchsorted(resident.cell_ids, packet.cell_ids), capacity - 1
    )
    present = (
        packet.cell_valid
        & resident.cell_valid[position]
        & (resident.cell_ids[position] == packet.cell_ids)
    )
    conflict = jnp.any(
        present
        & (
            (resident.cell_owner[position] != packet.cell_owner)
            | (resident.cell_classes[position] != packet.cell_classes)
            | jnp.any(resident.cell_exterior[position] != packet.cell_exterior, axis=1)
            | (resident.root_cell_ids[position] != packet.root_cell_ids)
            | jnp.any(
                resident.source_cell_ids[position] != packet.source_cell_ids, axis=1
            )
            | jnp.any(resident.cell_vertices[position] != packet.cell_vertices, axis=1)
            | jnp.any(
                resident.cell_coordinates[position] != packet.cell_coordinates,
                axis=(1, 2),
            )
            | jnp.any(
                resident.source_vertex_ids[position] != packet.source_vertex_ids,
                axis=(1, 2),
            )
            | jnp.any(
                resident.source_weights[position] != packet.source_weights, axis=(1, 2)
            )
        )
    )
    incoming = packet.cell_valid & adjacent & ~present
    valid = jnp.concatenate((resident.cell_valid, incoming))
    ids = jnp.where(
        valid,
        jnp.concatenate((resident.cell_ids, packet.cell_ids)),
        jnp.iinfo(jnp.int64).max,
    )
    order = jnp.argsort(ids, stable=True)
    count = jnp.sum(valid, dtype=jnp.int32)
    selected = order[:capacity]

    def merge(left: Array, right: Array) -> Array:
        return jnp.concatenate((left, right), axis=0)[selected]

    return SimplexNeighborhoodWorkset(
        ids[selected],
        merge(resident.cell_vertices, packet.cell_vertices),
        merge(resident.cell_coordinates, packet.cell_coordinates),
        merge(resident.cell_owner, packet.cell_owner),
        merge(resident.cell_classes, packet.cell_classes),
        merge(resident.cell_exterior, packet.cell_exterior),
        jnp.arange(capacity) < count,
        merge(resident.root_cell_ids, packet.root_cell_ids),
        merge(resident.source_cell_ids, packet.source_cell_ids),
        merge(resident.source_vertex_ids, packet.source_vertex_ids),
        merge(resident.source_weights, packet.source_weights),
        resident.status
        | packet.status
        | jnp.where(count > capacity, int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED), 0)
        | jnp.where(conflict, int(AdaptiveSimplexStatus.INVALID_GEOMETRY), 0),
    )


def _expand_local_neighborhood(
    parts: AdaptiveSimplexParts,
    work: SimplexNeighborhoodWorkset,
    halo_width: int,
    vertex_capacity: int,
    /,
) -> SimplexNeighborhoodWorkset:
    """One shared owner-local breadth-first packet/consensus protocol."""
    axis = parts.axis_name
    rank = jax.lax.axis_index(axis)

    def expand(
        _: int, previous: SimplexNeighborhoodWorkset
    ) -> SimplexNeighborhoodWorkset:
        frontier = _neighborhood_facet_keys(previous)
        current = previous
        for source, target in parts.neighbor_pairs:
            packet = jax.tree_util.tree_map(
                lambda value: jax.lax.ppermute(value, axis, ((source, target),)),
                previous,
            )
            packet = SimplexNeighborhoodWorkset(
                packet.cell_ids,
                packet.cell_vertices,
                packet.cell_coordinates,
                packet.cell_owner,
                packet.cell_classes,
                packet.cell_exterior,
                packet.cell_valid & (rank == target),
                packet.root_cell_ids,
                packet.source_cell_ids,
                packet.source_vertex_ids,
                packet.source_weights,
                jnp.where(rank == target, packet.status, 0),
            )
            current = _merge_neighborhood_packets(current, packet, frontier)
        return current

    work = jax.lax.fori_loop(0, halo_width, expand, work)
    ids = jnp.sort(
        jnp.where(
            work.cell_valid[:, None], work.cell_vertices, jnp.iinfo(jnp.int64).max
        ).reshape(-1)
    )
    fresh = (ids != jnp.iinfo(jnp.int64).max) & jnp.concatenate(
        (jnp.ones((1,), dtype=jnp.bool_), ids[1:] != ids[:-1])
    )
    overflow = jnp.sum(fresh, dtype=jnp.int32) > vertex_capacity
    status = work.status | jnp.where(
        overflow, int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED), 0
    )
    bits = (status >> jnp.arange(8, dtype=jnp.int32)) & 1
    status = jnp.sum(
        (jax.lax.psum(bits, axis) > 0).astype(jnp.int32)
        << jnp.arange(8, dtype=jnp.int32),
        dtype=jnp.int32,
    )
    return SimplexNeighborhoodWorkset(
        work.cell_ids,
        work.cell_vertices,
        work.cell_coordinates,
        work.cell_owner,
        work.cell_classes,
        work.cell_exterior,
        work.cell_valid,
        work.root_cell_ids,
        work.source_cell_ids,
        work.source_vertex_ids,
        work.source_weights,
        status,
    )


def _expand_simplex_neighborhood(
    parts: AdaptiveSimplexParts,
    states: AdaptiveSimplexState,
    initial_states: AdaptiveSimplexState,
    initial_exterior: Array,
    halo_width: int,
    cell_capacity: int,
    vertex_capacity: int,
    /,
) -> SimplexNeighborhoodWorkset:
    axis = parts.axis_name
    spec = PartitionSpec(axis)

    def local(
        state_block: AdaptiveSimplexState,
        initial_block: AdaptiveSimplexState,
        exterior_block: Array,
    ) -> SimplexNeighborhoodWorkset:
        state = jax.tree_util.tree_map(lambda value: value[0], state_block)
        initial = jax.tree_util.tree_map(lambda value: value[0], initial_block)
        rank = jax.lax.axis_index(axis)
        work = _simplex_neighborhood_seed(
            state, initial, exterior_block[0], rank, cell_capacity
        )

        work = _expand_local_neighborhood(parts, work, halo_width, vertex_capacity)
        return jax.tree_util.tree_map(lambda value: value[None], work)

    return jax.shard_map(
        local,
        mesh=parts.mesh,
        in_specs=(spec, spec, spec),
        out_specs=spec,
        check_vma=False,
    )(states, initial_states, initial_exterior)


_compiled_neighborhood_expansion = eqx.filter_jit(_expand_simplex_neighborhood)


def _expand_existing_simplex_neighborhood(
    parts: AdaptiveSimplexParts,
    workset: SimplexNeighborhoodWorkset,
    halo_width: int,
    vertex_capacity: int,
    /,
) -> SimplexNeighborhoodWorkset:
    spec = PartitionSpec(parts.axis_name)

    def local(block: SimplexNeighborhoodWorkset) -> SimplexNeighborhoodWorkset:
        work = jax.tree_util.tree_map(lambda value: value[0], block)
        work = _expand_local_neighborhood(parts, work, halo_width, vertex_capacity)
        return jax.tree_util.tree_map(lambda value: value[None], work)

    return jax.shard_map(
        local, mesh=parts.mesh, in_specs=(spec,), out_specs=spec, check_vma=False
    )(workset)


_compiled_existing_neighborhood_expansion = eqx.filter_jit(
    _expand_existing_simplex_neighborhood
)


def expand_simplex_neighborhood_workset(
    parts: AdaptiveSimplexParts,
    workset: SimplexNeighborhoodWorkset,
    /,
    *,
    halo_width: int,
    vertex_capacity: int,
) -> SimplexNeighborhoodWorkset:
    """Rebuild actual ghosts of migrated packets without discarding their ancestry."""
    width = operator.index(halo_width)
    vertices = operator.index(vertex_capacity)
    if (
        isinstance(halo_width, bool)
        or isinstance(vertex_capacity, bool)
        or width < 0
        or vertices <= 0
        or workset.cell_ids.ndim != 2
        or workset.cell_ids.shape[0] != parts.part_count
        or workset.cell_valid.shape != workset.cell_ids.shape
    ):
        raise ValueError(
            "Migrated neighborhood worksets require valid placement and bounded capacities."
        )
    return _compiled_existing_neighborhood_expansion(parts, workset, width, vertices)


def expand_partitioned_simplex_neighborhood(
    parts: AdaptiveSimplexParts,
    states: AdaptiveSimplexState,
    initial_states: AdaptiveSimplexState,
    initial_exterior: Array,
    /,
    *,
    halo_width: int,
    cell_capacity: int,
    vertex_capacity: int,
) -> SimplexNeighborhoodWorkset:
    """Expand actual neighboring cells by bounded packets, with collective refusal."""
    width = operator.index(halo_width)
    cells = operator.index(cell_capacity)
    vertices = operator.index(vertex_capacity)
    if (
        any(
            isinstance(value, bool)
            for value in (halo_width, cell_capacity, vertex_capacity)
        )
        or width < 0
        or cells <= 0
        or vertices <= 0
        or states.mesh.cells.shape != initial_states.mesh.cells.shape
        or states.mesh.cells.shape[0] != parts.part_count
        or initial_exterior.shape != states.mesh.cells.shape
        or initial_exterior.dtype != jnp.bool_
    ):
        raise ValueError(
            "Neighborhood expansion requires matched states and bounded valid capacities."
        )
    return _compiled_neighborhood_expansion(
        parts, states, initial_states, initial_exterior, width, cells, vertices
    )


def _owner_local_partition_summary(mesh: CellMesh, /) -> tuple[Array, Array, Array]:
    """Reduce actual canonical owner/facet/ghost arrays to bounded summaries."""
    storage = mesh.storage
    if storage is None:
        raise ValueError("Collective partition summaries require owner-local storage.")
    arrays = dict(storage.logical_arrays)
    required = {
        "cell_owners",
        "cell_vertices",
        "closure/cell_owner",
        "closure/cell_valid",
    }
    if not required.issubset(arrays):
        raise ValueError(
            "Owner-local publication lacks its actual placement/closure arrays."
        )
    owners = arrays["cell_owners"]
    vertices = arrays["cell_vertices"]
    active = (
        jnp.arange(owners.shape[0], dtype=jnp.int64) < storage.global_entity_counts[-1]
    )
    weights = jnp.bincount(
        owners,
        weights=active.astype(jnp.float64),
        length=storage.partition_count,
    )
    width = vertices.shape[1]
    columns = jnp.asarray(_facet_columns(width), dtype=jnp.int32)
    facets = jnp.sort(vertices[:, columns], axis=2).reshape((-1, width - 1))
    facets = jnp.where(
        jnp.repeat(active, width)[:, None], facets, jnp.iinfo(jnp.int64).max
    )
    order = jnp.lexsort(tuple(facets[:, column] for column in range(width - 2, -1, -1)))
    facets = facets[order]
    facet_owners = jnp.repeat(owners, width)[order]
    shared = (
        jnp.all(facets[1:] == facets[:-1], axis=1)
        & (facets[1:, 0] != jnp.iinfo(jnp.int64).max)
        & (facet_owners[1:] != facet_owners[:-1])
    )
    edge_cut = jnp.sum(shared, dtype=jnp.int64)
    halo_replicas = jnp.sum(
        arrays["closure/cell_valid"]
        & (
            arrays["closure/cell_owner"]
            != jnp.arange(storage.partition_count, dtype=jnp.int32)[:, None]
        ),
        dtype=jnp.int64,
    )
    return weights, edge_cut, halo_replicas


def prepare_owner_local_distribution_transition(
    source: MeshDistribution,
    target_part: MeshPart,
    lineage: MeshLineage,
    source_rows: ArrayLike,
    target_rows: ArrayLike,
    source_cell_ids: ArrayLike,
    source_cell_owners: ArrayLike,
    /,
    *,
    policy: MeshPartitionPolicy,
    restart_repack: SimplexRestartRepack | None = None,
    restart_proof: SimplexRestartRepackProof | None = None,
) -> MeshDistributionTransition:
    """Publish retained owner routes only after actual halo/balance consensus.

    Source caches are validated against accepted identity, avoiding a complete
    source-array host transfer during commit. The bounded summary is global;
    transport slots and migration counts remain explicitly owner-local.
    """
    if not isinstance(target_part.carrier, CellMeshingResult):
        raise ValueError(
            "Owner-local distribution transitions require a certified local target."
        )
    mesh = target_part.carrier.mesh
    storage = mesh.storage
    if storage is None:
        raise ValueError(
            "Owner-local distribution transitions require a certified local target."
        )
    if (
        not isinstance(policy, MeshPartitionPolicy)
        or (restart_repack is None and policy.part_count != source.partition.part_count)
        or (restart_repack is None and policy.kind is not source.evidence.kind)
        or policy.part_count != storage.partition_count
        or policy.halo_width > storage.neighborhood_depth
    ):
        raise ValueError(
            "Retained ownership must preserve its actual partition algorithm and meet "
            "the requested owner count/neighborhood; repartitioning requires a validated migration."
        )
    if restart_repack is not None:
        graph = restart_repack.graph_proposal
        executed = (
            MeshPartitionKind.PROVIDER if graph is None else MeshPartitionKind.GRAPH
        )
        if policy.kind is not executed or (
            graph is not None and graph.maximum_imbalance != policy.maximum_imbalance
        ):
            raise ValueError(
                "Restart ownership must name the route that produced its validated proposal: "
                "GRAPH only for a replayed native graph partition under the same plan, "
                "PROVIDER for an explicit requested placement."
            )
    summary = _owner_local_partition_summary(mesh)
    target = MeshDistribution(
        target_part,
        CellPartition(storage.entity_owner[-1], storage.partition_count, storage=storage),
        halo_width=policy.halo_width,
        partition_kind=policy.kind,
        partition_provenance=(
            "collective-retained-ownership"
            if restart_repack is None
            else "collective-validated-restart-ownership"
            if restart_repack.graph_proposal is None
            else "collective-replayed-native-graph-restart-ownership"
        ),
        _collective_summary=summary,
    )
    if (
        np.asarray(jax.device_get(target.evidence.imbalance)).item()
        > policy.maximum_imbalance
    ):
        raise ValueError(
            "Requested balance requires native repartition plus complete epoch-forest migration; "
            "retained ownership cannot satisfy this policy."
        )
    transition = MeshDistributionTransition(
        source,
        target,
        source_rows,
        target_rows,
        lineage,
        rebalanced=restart_repack is not None,
        _source_cell_ids=source_cell_ids,
        _source_cell_owners=source_cell_owners,
        restart_repack=restart_repack,
        restart_proof=restart_proof,
    )
    if restart_repack is None and np.any(
        np.asarray(transition.message_source_ranks)
        != np.asarray(transition.message_target_ranks)
    ):
        raise ValueError(
            "Retained owner publication cannot silently perform an unprepared migration."
        )
    return transition


__all__ = [
    "MeshDistribution",
    "MeshDistributionTransition",
    "MeshPartitionEvidence",
    "MeshPartitionKind",
    "MeshPartitionPolicy",
    "AffineBisectionSourceWitness",
    "SimplexNeighborhoodWorkset",
    "expand_partitioned_simplex_neighborhood",
    "expand_simplex_neighborhood_workset",
    "prepare_distribution_transition",
    "prepare_owner_local_distribution_transition",
    "prepare_mesh_distribution",
]

# Checked restart annotations resolve at runtime. The restart owner consumes
# this module's workset/policy types, so it binds only after their definitions.
from ._restart_distribution import (
    SimplexRestartRepack,
    SimplexRestartRepackProof,
)
