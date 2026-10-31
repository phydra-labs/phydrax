#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Mesh ownership, ghost residency, and ownership migration across revisions.

Ownership comes from weighted contiguous ranges along Morton or Hilbert
space-filling curves, native multilevel graph partitioning, an explicitly
selected METIS comparison, or a provider. Ghost layers are breadth-first
face-adjacency reach. Transitions carry ownership through mesh lineage onto
adapted parts, repartition only when inherited ownership is out of balance,
and migrate cell data through explicit send/receive sparse relations.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from enum import StrEnum
from math import prod
from typing import Any, final, TYPE_CHECKING, TypedDict

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._execution_runtime import ExecutionGroup
from .._fingerprint import (
    array_tree_fingerprint,
    canonical_array_hex_fingerprint,
    canonical_fingerprint,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellPartition, PointCloudPlan, PreparedTensorGrid
from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_geometry import (
    CellGeometryCoordinateMap,
    PerCellRestrictedCellGeometryElement,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..discretization._distributed_field import DistributedHaloPlan
from ..discretization._partition import (
    CellAdjacency,
    CellPartitionHalo,
    inherit_cell_owners,
    mesh_cell_adjacency,
    padded_part_table,
)
from ..discretization._periodic_cell import PeriodicCell
from ..discretization._periodic_exact import identification_orders
from ..discretization._periodic_topology import _entity_vertex_rows
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
from ..typing import checked, parse
from ._assembly import MeshPart
from ._lineage import MeshLineage
from ._metis import metis_identity, metis_partition, MetisPartitionError
from ._periodic import (
    _periodic_cell_layout,
    _periodic_collapse_owners,
    _periodic_expand_owners,
    _periodic_lifted_adjacency,
    _periodic_quotient_adjacency,
    _periodic_quotient_cell_centroids,
    _periodic_quotient_weights,
    _PeriodicCellLayout,
    periodic_orbit_closure,
)
from ._result import CellMeshingResult


if TYPE_CHECKING:
    from ..graph._distributed_partition import DistributedGraphPartitionResult
    from ..lifecycle import (
        Composition,
        CompositionEntry,
        CompositionRebindReceipt,
        CompositionRole,
        CompositionTransport,
    )
    from ._assembly import MeshAssembly
    from ._periodic import OwnerLocalPeriodicCandidate


_OWNER_LOCAL_IDENTITY_BUCKET_COUNT = 256
_MAXIMUM_CURVE_DEPTH = 21


class _LocalCellGeometryBlock(TypedDict):
    name: str
    cell_kind: str
    cell_rows: np.ndarray
    cell_ids: np.ndarray
    vertices: np.ndarray
    vertex_valid: np.ndarray
    coordinate_routes: np.ndarray
    element: object


class _LocalCellGeometry(TypedDict):
    part_id: str
    distribution_id: str
    cell_rows: np.ndarray
    cell_ids: np.ndarray
    owned: np.ndarray
    vertex_rows: np.ndarray
    vertex_ids: np.ndarray
    vertex_coordinates: np.ndarray
    coordinate_rows: np.ndarray
    coordinates: np.ndarray
    geometry_layout_id: str
    blocks: tuple[_LocalCellGeometryBlock, ...]


def _periodic_mesh(part: MeshPart, /) -> CellMesh | None:
    carrier = part.carrier
    return carrier.mesh if isinstance(carrier, CellMeshingResult) else None


class MeshPartitionKind(StrEnum):
    """Ownership route: curve, native graph, explicit METIS comparison, or provider."""

    MORTON = "morton"
    HILBERT = "hilbert"
    GRAPH = "graph"
    METIS = "metis"
    PROVIDER = "provider"


class MeshPartitionPolicy(StrictModule, NonTrainableState):
    """How cell ownership is produced, balanced, and ghosted.

    ``maximum_imbalance`` is the requested maximum part-weight / mean-part-weight
    target for graph routes and the target whose measured feasibility is reported
    for every route. Curve routes also retain ``curve_depth``; only the explicit
    METIS comparison uses ``graph_seed``. Equal curve codes are ordered by stable
    global cell ID.

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
    ) -> None:
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
        if (
            kind not in (MeshPartitionKind.MORTON, MeshPartitionKind.HILBERT)
            and depth != 16
        ):
            raise ValueError("curve_depth applies only to Morton and Hilbert routes.")
        if kind is not MeshPartitionKind.METIS and seed != 0:
            raise ValueError(
                "graph_seed applies only to the explicit METIS comparison route."
            )
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
                "curve_depth": depth
                if kind in (MeshPartitionKind.MORTON, MeshPartitionKind.HILBERT)
                else None,
                "graph_seed": seed if kind is MeshPartitionKind.METIS else None,
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
    feasible: bool = eqx.field(static=True)
    feasibility_status: str = eqx.field(static=True)
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
        feasible: bool,
        /,
    ) -> None:
        part_weights = np.bincount(owner, weights=weights, minlength=part_count)
        imbalance = np.max(part_weights) / np.mean(part_weights)
        pairs = adjacency.undirected_pairs()
        edge_cut = np.count_nonzero(owner[pairs[:, 0]] != owner[pairs[:, 1]])
        self.part_weights = jnp.asarray(part_weights, dtype=jnp.float64)
        self.imbalance = jnp.asarray(imbalance, dtype=jnp.float64)
        self.edge_cut = jnp.asarray(edge_cut, dtype=jnp.int64)
        self.halo_replicas = jnp.asarray(halo_replicas, dtype=jnp.int64)
        self.feasible = bool(feasible)
        self.feasibility_status = "feasible" if feasible else "balance_not_reached"
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
                "feasible": bool(feasible),
            }
        )


def _native_cell_ids(part: MeshPart) -> np.ndarray:
    from ..discretization._view_support import OwnerLocalCellGeometry

    carrier = part.carrier
    if isinstance(carrier, CellMeshingResult):
        return np.concatenate(
            tuple(np.asarray(block.global_ids) for block in carrier.mesh.blocks)
        ).astype(np.int64)
    if isinstance(carrier, PreparedTensorGrid):
        return np.arange(prod(carrier.cells().shape), dtype=np.int64)
    if isinstance(carrier, PointCloudPlan):
        return np.arange(carrier.points.shape[0], dtype=np.int64)
    if isinstance(carrier, IsogeometricPlan):
        return np.arange(prod(carrier.topology.axis_sizes), dtype=np.int64)
    if isinstance(carrier, OwnerLocalCellGeometry):
        raise TypeError(
            "Owner-local cell views require their collective sharded distribution."
        )
    raise TypeError("Unsupported mesh carrier for global MeshDistribution.")


def _periodic_layout(
    part: MeshPart, /
) -> tuple[CellMesh | None, _PeriodicCellLayout | None]:
    mesh = _periodic_mesh(part)
    return mesh, None if mesh is None else _periodic_cell_layout(mesh)


def _require_periodic_mesh(mesh: CellMesh | None, /) -> CellMesh:
    if mesh is None:
        raise RuntimeError("Periodic quotient layout lost its source mesh.")
    return mesh


def _rows_of(native: np.ndarray, identifiers: ArrayLike, /) -> np.ndarray:
    """Native rows of ``identifiers``; -1 where an ID is not a native cell."""
    values = np.asarray(identifiers, dtype=np.int64).reshape(-1)
    sorter = np.argsort(native, kind="stable")
    located = np.minimum(np.searchsorted(native, values, sorter=sorter), native.size - 1)
    rows = sorter[located]
    return np.where(native[rows] == values, rows, -1)


def _periodic_support_owner_routes(
    support_fractional_lower: ArrayLike,
    support_fractional_upper: ArrayLike,
    owner_fractional_bounds: tuple[tuple[ArrayLike, ArrayLike] | None, ...],
    periodic_axes: tuple[bool, ...],
    /,
) -> tuple[np.ndarray, ...]:
    """Route circumball witnesses only to owners whose points may intersect.

    Both AABBs are outward enclosures in fractional coordinates. Periodic axes
    admit every integer deck translation; outward-rounded subtraction before
    ``ceil``/``floor`` keeps routing conservative without a tolerance.
    """
    lower = np.asarray(support_fractional_lower, dtype=np.float64)
    upper = np.asarray(support_fractional_upper, dtype=np.float64)
    axes = tuple(periodic_axes)
    dimension = len(axes)
    if (
        lower.ndim != 2
        or lower.shape[1] != dimension
        or upper.shape != lower.shape
        or dimension not in (2, 3)
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower > upper)
        or any(not isinstance(axis, bool) for axis in axes)
        or not owner_fractional_bounds
    ):
        raise ValueError("Periodic support and owner bounds are malformed.")
    owners: list[tuple[np.ndarray, np.ndarray] | None] = []
    for bounds in owner_fractional_bounds:
        if bounds is None:
            owners.append(None)
            continue
        owner_lower = np.asarray(bounds[0], dtype=np.float64)
        owner_upper = np.asarray(bounds[1], dtype=np.float64)
        if (
            owner_lower.shape != (dimension,)
            or owner_upper.shape != (dimension,)
            or not np.all(np.isfinite(owner_lower))
            or not np.all(np.isfinite(owner_upper))
            or np.any(owner_lower > owner_upper)
        ):
            raise ValueError("An owner fractional AABB is malformed.")
        owners.append((owner_lower, owner_upper))

    routes: list[np.ndarray] = []
    for owner_bounds in owners:
        if owner_bounds is None:
            routes.append(np.zeros((0,), dtype=np.int64))
            continue
        owner_lower, owner_upper = owner_bounds
        selected = np.ones((lower.shape[0],), dtype=np.bool_)
        for axis, periodic in enumerate(axes):
            if periodic:
                difference_lower = np.nextafter(
                    lower[:, axis] - owner_upper[axis], -np.inf
                )
                difference_upper = np.nextafter(
                    upper[:, axis] - owner_lower[axis], np.inf
                )
                finite = np.isfinite(difference_lower) & np.isfinite(difference_upper)
                possible = np.ones((lower.shape[0],), dtype=np.bool_)
                if np.any(finite):
                    lower_integer = np.ceil(difference_lower[finite])
                    upper_integer = np.floor(difference_upper[finite])
                    possible[finite] = lower_integer <= upper_integer
                selected &= possible
            else:
                selected &= (lower[:, axis] <= owner_upper[axis]) & (
                    upper[:, axis] >= owner_lower[axis]
                )
        routes.append(np.flatnonzero(selected).astype(np.int64, copy=False))
    return tuple(routes)


def _periodic_fractional_bounds_from_cartesian(
    lower: ArrayLike,
    upper: ArrayLike,
    origin: ArrayLike,
    lattice: ArrayLike,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Outward exact-binary fractional enclosure of a Cartesian box."""
    from fractions import Fraction

    from ..geometry._rational import fraction_down, fraction_up
    from .providers._native_periodic_power import _exact_lattice_inverse

    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    origin_ = np.asarray(origin, dtype=np.float64)
    vectors = np.asarray(lattice, dtype=np.float64)
    dimension = origin_.size
    if (
        low.shape != (dimension,)
        or high.shape != (dimension,)
        or vectors.shape != (dimension, dimension)
        or not np.all(np.isfinite(low))
        or not np.all(np.isfinite(high))
        or not np.all(np.isfinite(origin_))
        or not np.all(np.isfinite(vectors))
        or np.any(low > high)
    ):
        raise ValueError("A full-rank periodic Cartesian AABB is malformed.")
    inverse = _exact_lattice_inverse(vectors)
    exact_origin = tuple(Fraction.from_float(float(value)) for value in origin_)
    exact_low = tuple(
        Fraction.from_float(float(value)) - exact_origin[axis]
        for axis, value in enumerate(low)
    )
    exact_high = tuple(
        Fraction.from_float(float(value)) - exact_origin[axis]
        for axis, value in enumerate(high)
    )
    fractional_lower = []
    fractional_upper = []
    for coordinate in range(dimension):
        lower_sum = Fraction(0)
        upper_sum = Fraction(0)
        for axis in range(dimension):
            coefficient = inverse[axis][coordinate]
            first, second = exact_low[axis] * coefficient, exact_high[axis] * coefficient
            lower_sum += min(first, second)
            upper_sum += max(first, second)
        fractional_lower.append(fraction_down(lower_sum))
        fractional_upper.append(fraction_up(upper_sum))
    return (
        np.asarray(fractional_lower, dtype=np.float64),
        np.asarray(fractional_upper, dtype=np.float64),
    )


def _periodic_fractional_point_bounds(
    points: ArrayLike,
    origin: ArrayLike,
    lattice: ArrayLike,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact-outward fractional intervals for a local source-point bank."""
    from fractions import Fraction

    from ..geometry._rational import fraction_down, fraction_up
    from .providers._native_periodic_power import _exact_lattice_inverse

    values = np.asarray(points, dtype=np.float64)
    origin_ = np.asarray(origin, dtype=np.float64)
    vectors = np.asarray(lattice, dtype=np.float64)
    dimension = origin_.size
    if (
        values.ndim != 2
        or values.shape[1] != dimension
        or vectors.shape != (dimension, dimension)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(origin_))
    ):
        raise ValueError("Periodic source points do not match their lattice frame.")
    if values.shape[0] == 0:
        empty = np.zeros((0, dimension), dtype=np.float64)
        return empty, empty.copy()
    inverse = _exact_lattice_inverse(vectors)
    exact_origin = tuple(Fraction.from_float(float(value)) for value in origin_)
    lower = np.empty((values.shape[0], dimension), dtype=np.float64)
    upper = np.empty_like(lower)
    for row in range(values.shape[0]):
        delta = tuple(
            Fraction.from_float(float(values[row, axis])) - exact_origin[axis]
            for axis in range(dimension)
        )
        for coordinate in range(dimension):
            fractional = sum(
                (delta[axis] * inverse[axis][coordinate] for axis in range(dimension)),
                Fraction(0),
            )
            lower[row, coordinate] = fraction_down(fractional)
            upper[row, coordinate] = fraction_up(fractional)
    return lower, upper


def _periodic_power_support_boxes(
    points: ArrayLike,
    origin: ArrayLike,
    lattice: ArrayLike,
    diameter: float,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Outward fractional AABBs of Cartesian-radius neighborhoods around sites."""
    from fractions import Fraction

    from ..geometry._rational import fraction_down, fraction_up
    from .providers._native_periodic_power import (
        _exact_lattice_inverse,
        _upper_sqrt,
    )

    values = np.asarray(points, dtype=np.float64)
    origin_ = np.asarray(origin, dtype=np.float64)
    vectors = np.asarray(lattice, dtype=np.float64)
    dimension = origin_.size
    radius = float(diameter)
    if (
        values.ndim != 2
        or values.shape[1] != dimension
        or vectors.shape != (dimension, dimension)
        or not np.all(np.isfinite(values))
        or not np.isfinite(radius)
        or radius <= 0.0
    ):
        raise ValueError("Periodic power support sites/radius do not match the lattice.")
    if values.shape[0] == 0:
        empty = np.zeros((0, dimension), dtype=np.float64)
        return empty, empty.copy()
    inverse = _exact_lattice_inverse(vectors)
    exact_origin = tuple(Fraction.from_float(float(value)) for value in origin_)
    radius_squared = Fraction.from_float(radius) ** 2
    axis_radii = tuple(
        Fraction.from_float(
            _upper_sqrt(
                radius_squared
                * sum(
                    (inverse[row][axis] ** 2 for row in range(dimension)),
                    Fraction(0),
                )
            )
        )
        for axis in range(dimension)
    )
    lower = np.empty((values.shape[0], dimension), dtype=np.float64)
    upper = np.empty_like(lower)
    for row in range(values.shape[0]):
        delta = tuple(
            Fraction.from_float(float(values[row, axis])) - exact_origin[axis]
            for axis in range(dimension)
        )
        fractional = tuple(
            sum(
                (delta[axis] * inverse[axis][coordinate] for axis in range(dimension)),
                Fraction(0),
            )
            for coordinate in range(dimension)
        )
        for axis in range(dimension):
            lower[row, axis] = fraction_down(fractional[axis] - axis_radii[axis])
            upper[row, axis] = fraction_up(fractional[axis] + axis_radii[axis])
    return lower, upper


def _periodic_support_point_rows(
    support_lower: ArrayLike,
    support_upper: ArrayLike,
    point_lower: ArrayLike,
    point_upper: ArrayLike,
    periodic_axes: tuple[bool, ...],
    /,
) -> np.ndarray:
    """Owner-local source rows whose periodic images intersect any support box."""
    lower = np.asarray(support_lower, dtype=np.float64)
    upper = np.asarray(support_upper, dtype=np.float64)
    point_low = np.asarray(point_lower, dtype=np.float64)
    point_high = np.asarray(point_upper, dtype=np.float64)
    axes = tuple(periodic_axes)
    dimension = len(axes)
    if (
        lower.ndim != 2
        or lower.shape[1] != dimension
        or upper.shape != lower.shape
        or point_low.ndim != 2
        or point_low.shape[1] != dimension
        or point_high.shape != point_low.shape
        or np.any(lower > upper)
        or np.any(point_low > point_high)
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or not np.all(np.isfinite(point_low))
        or not np.all(np.isfinite(point_high))
        or any(not isinstance(axis, bool) for axis in axes)
    ):
        raise ValueError(
            "Periodic support boxes and source point intervals are malformed."
        )
    found = np.zeros((point_low.shape[0],), dtype=np.bool_)
    chunk_width = 256
    for start in range(0, lower.shape[0], chunk_width):
        stop = min(start + chunk_width, lower.shape[0])
        possible = np.ones((stop - start, point_low.shape[0]), dtype=np.bool_)
        for axis, periodic in enumerate(axes):
            if periodic:
                difference_lower = np.nextafter(
                    lower[start:stop, None, axis] - point_high[None, :, axis],
                    -np.inf,
                )
                difference_upper = np.nextafter(
                    upper[start:stop, None, axis] - point_low[None, :, axis],
                    np.inf,
                )
                finite = np.isfinite(difference_lower) & np.isfinite(difference_upper)
                axis_possible = np.ones(possible.shape, dtype=np.bool_)
                if np.any(finite):
                    lower_integer = np.ceil(difference_lower[finite])
                    upper_integer = np.floor(difference_upper[finite])
                    axis_possible[finite] = lower_integer <= upper_integer
                possible &= axis_possible
            else:
                possible &= (
                    lower[start:stop, None, axis] <= point_high[None, :, axis]
                ) & (upper[start:stop, None, axis] >= point_low[None, :, axis])
        found |= np.any(possible, axis=0)
    return np.flatnonzero(found).astype(np.int64, copy=False)


def _owner_group_allgather(
    local_values: ArrayLike, execution_group: Any, axis_name: str, /
) -> np.ndarray:
    """Allgather only compact owner metadata on the registered execution group."""
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from .._execution_runtime import ExecutionGroup

    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    devices = execution_group.devices
    group_ranks = execution_group.spec.process_indices
    axis = str(axis_name).strip()
    if (
        not axis
        or len(devices) != len(group_ranks)
        or tuple(device.process_index for device in devices) != group_ranks
        or not execution_group.is_member
    ):
        raise ValueError(
            "Owner metadata collectives require one group device per member process."
        )
    values = np.asarray(local_values)
    if values.ndim < 1 or values.dtype.hasobject:
        raise TypeError("Owner metadata must be a non-object array with a leading axis.")
    mesh = Mesh(np.asarray(devices, dtype=object), (axis,))
    partition = PartitionSpec(axis)
    distributed = jax.make_array_from_process_local_data(
        NamedSharding(mesh, partition),
        values[None, ...],
        global_shape=(len(devices), *values.shape),
    )

    def gather(local: Array, /) -> Array:
        return jax.lax.all_gather(local[0], axis, axis=0, tiled=False)

    gathered = jax.shard_map(
        gather,
        mesh=mesh,
        in_specs=partition,
        out_specs=PartitionSpec(),
        check_vma=False,
    )(distributed)
    return np.asarray(jax.device_get(gathered))


def _owner_pair_channel_plan(
    sender: int, receiver: int, part_count: int, /
) -> DistributedHaloPlan:
    """One sparse directed owner message and its reverse-reply row."""
    if (
        isinstance(sender, bool)
        or isinstance(receiver, bool)
        or isinstance(part_count, bool)
        or not 0 <= sender < part_count
        or not 0 <= receiver < part_count
        or sender == receiver
    ):
        raise ValueError("Owner channel endpoints must be distinct valid ranks.")
    return DistributedHaloPlan(
        np.asarray((sender, receiver), dtype=np.int32),
        np.asarray(((0, 1),), dtype=np.int32),
        part_count,
    )


def _owner_exchange_message_banks(
    outgoing_by_owner: tuple[np.ndarray, ...],
    local_rank: int,
    execution_group: Any,
    axis_name: str,
    /,
) -> dict[int, np.ndarray]:
    """Exchange only local per-owner messages, one directed owner pair at a time."""
    if not outgoing_by_owner:
        raise ValueError("Owner message exchange requires at least one rank.")
    parts = len(outgoing_by_owner)
    if (
        isinstance(local_rank, bool)
        or not isinstance(local_rank, (int, np.integer))
        or not 0 <= int(local_rank) < parts
    ):
        raise ValueError("Owner message rank is outside the communication group.")
    rank = int(local_rank)
    banks = tuple(np.asarray(value) for value in outgoing_by_owner)
    width = banks[0].shape[1] if banks[0].ndim == 2 else -1
    dtype = banks[0].dtype
    if (
        width < 1
        or dtype.hasobject
        or any(
            value.ndim != 2 or value.shape[1] != width or value.dtype != dtype
            for value in banks
        )
        or banks[rank].shape[0] != 0
    ):
        raise ValueError("Per-owner message banks must share a typed row layout.")
    local_counts = np.asarray([value.shape[0] for value in banks], dtype=np.int64)
    count_matrix = _owner_group_allgather(local_counts, execution_group, axis_name)
    if count_matrix.shape != (parts, parts) or np.any(count_matrix < 0):
        raise ValueError("Collective owner message counts are malformed.")
    if np.any(np.diag(count_matrix)):
        raise ValueError("Owner-local messages must not route through self channels.")
    if not np.array_equal(count_matrix[rank], local_counts):
        raise ValueError("Collective message counts disagree with this owner shard.")
    incoming: dict[int, np.ndarray] = {}
    for sender_value, receiver_value in np.argwhere(count_matrix > 0):
        sender, receiver = int(sender_value), int(receiver_value)
        count = int(count_matrix[sender, receiver])
        plan = _owner_pair_channel_plan(sender, receiver, parts)
        local_ids = np.asarray(plan.local_global_ids, dtype=np.int64)[rank]
        local_valid = np.asarray(plan.local_valid, dtype=np.bool_)[rank]
        slots = {
            int(identifier): row
            for row, identifier in enumerate(local_ids)
            if local_valid[row]
        }
        local_values = np.zeros((plan.local_capacity, count, width), dtype=dtype)
        if rank == sender:
            if banks[receiver].shape[0] != count:
                raise ValueError("Owner message rows differ from the collective count.")
            local_values[slots[0]] = banks[receiver]
        exchanged = _owner_channel_exchange(
            plan, local_values, execution_group, axis_name
        )
        if rank == receiver:
            incoming[sender] = exchanged[slots[0]].copy()
    return incoming


def _owner_group_minimum_owner_ranks(
    query_ids: ArrayLike,
    owner_candidates: ArrayLike,
    local_rank: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> np.ndarray:
    """Elect the minimum incident-cell owner through hash-sharded ID records."""
    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    if isinstance(local_rank, (bool, np.bool_)) or not isinstance(
        local_rank, (int, np.integer)
    ):
        raise TypeError("local_rank must be an integer group slot.")
    rank = int(local_rank)
    parts = len(execution_group.spec.process_indices)
    query = np.asarray(query_ids)
    candidates = np.asarray(owner_candidates)
    local_error = None
    if (
        not isinstance(axis_name, str)
        or not axis_name.strip()
        or not 0 <= rank < parts
        or query.ndim != 1
        or query.dtype.kind not in "iu"
        or np.any(query < 0)
        or (
            query.dtype.kind == "u"
            and query.size
            and int(np.max(query)) > np.iinfo(np.int64).max
        )
        or np.unique(query).size != query.size
        or candidates.ndim != 2
        or candidates.shape[1] != 2
        or candidates.dtype.kind not in "iu"
        or np.any(candidates < 0)
        or (
            candidates.dtype.kind == "u"
            and candidates.size
            and int(np.max(candidates)) > np.iinfo(np.int64).max
        )
        or np.any(candidates[:, 1] >= parts)
    ):
        local_error = "Periodic owner-rank election inputs are malformed."
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        axis_name,
        "periodic owner-rank election input",
    )
    query = query.astype(np.int64, copy=False)
    candidates = candidates.astype(np.int64, copy=False)
    buckets = _owner_directory_bucket(candidates[:, 0], parts)
    local_candidates = candidates[buckets == rank]
    outgoing = tuple(
        candidates[(buckets == destination) & (destination != rank)].astype(
            np.int64, copy=False
        )
        if destination != rank
        else np.zeros((0, 2), dtype=np.int64)
        for destination in range(parts)
    )
    incoming = _owner_exchange_message_banks(
        outgoing, rank, execution_group, f"{axis_name}_candidates"
    )
    owner_by_id: dict[int, int] = {}
    local_error = None
    try:
        candidate_banks = (
            local_candidates,
            *(np.asarray(bank, dtype=np.int64) for bank in incoming.values()),
        )
        for bank in candidate_banks:
            if bank.ndim != 2 or bank.shape[1] != 2:
                raise ValueError(
                    "Periodic owner candidate bank has a malformed row layout."
                )
            for identifier_value, owner_value in bank.tolist():
                identifier, owner = int(identifier_value), int(owner_value)
                if identifier < 0 or not 0 <= owner < parts:
                    raise ValueError(
                        "Periodic owner candidate bank contains invalid labels."
                    )
                owner_by_id[identifier] = min(owner_by_id.get(identifier, owner), owner)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_reduce",
        "periodic owner-rank reduction",
    )
    shard_ids = np.asarray(sorted(owner_by_id), dtype=np.int64)
    shard_owners = np.asarray(
        [owner_by_id[int(identifier)] for identifier in shard_ids],
        dtype=np.int32,
    )
    return _owner_directory_lookup(
        query,
        shard_ids,
        shard_owners,
        rank,
        parts,
        execution_group,
        f"{axis_name}_lookup",
        "periodic vertex owner",
    )


def _owner_local_multiset_identity(
    kind: str,
    header: object,
    local_records: tuple[tuple[object, object], ...],
    source_id: str,
    source_revision: str | None,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> str:
    """Fingerprint stable owner-sharded records through fixed logical buckets."""
    import jax

    if (
        not isinstance(kind, str)
        or not kind
        or not isinstance(source_id, str)
        or source_revision is not None
        and (not isinstance(source_revision, str) or not source_revision)
        or not isinstance(execution_group, ExecutionGroup)
        or not isinstance(axis_name, str)
        or not axis_name.strip()
    ):
        raise ValueError("Distributed identity requires complete scientific inputs.")
    parts = len(execution_group.spec.process_indices)
    identity_header: dict[str, object] = {
        "kind": kind,
        "source_id": source_id,
        "header": header,
    }
    if source_revision is not None:
        identity_header["source_revision"] = source_revision
    header_digest = canonical_fingerprint(identity_header)
    headers = _owner_group_allgather(
        np.frombuffer(bytes.fromhex(header_digest), dtype=np.uint8),
        execution_group,
        f"{axis_name}_header",
    )
    if any(
        np.asarray(value, dtype=np.uint8).tobytes().hex() != header_digest
        for value in headers
    ):
        raise ValueError("Distributed identity declarations differ across ranks.")
    rank = execution_group.spec.process_indices.index(int(jax.process_index()))
    outgoing_rows: list[list[np.ndarray]] = [[] for _ in range(parts)]
    local_rows: list[np.ndarray] = []
    local_error = None
    try:
        for key, record in local_records:
            key_digest = bytes.fromhex(canonical_fingerprint(key))
            record_digest = bytes.fromhex(
                canonical_fingerprint(
                    {
                        "kind": kind,
                        "header": header_digest,
                        "key": key,
                        "record": record,
                    }
                )
            )
            bucket = key_digest[0]
            row = np.frombuffer(
                bytes((bucket,)) + key_digest + record_digest,
                dtype=np.uint8,
            ).copy()
            destination = bucket % parts
            if destination == rank:
                local_rows.append(row)
            else:
                outgoing_rows[destination].append(row)
    except Exception as error:
        local_error = str(error)
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_encode",
        "distributed identity record encoding",
    )
    outgoing = tuple(
        np.stack(rows) if rows else np.zeros((0, 65), dtype=np.uint8)
        for rows in outgoing_rows
    )
    incoming = _owner_exchange_message_banks(
        outgoing, rank, execution_group, f"{axis_name}_records"
    )
    entries: dict[int, list[tuple[bytes, bytes]]] = {}
    for row in (
        *local_rows,
        *(
            np.asarray(bank, dtype=np.uint8)[index]
            for bank in incoming.values()
            for index in range(np.asarray(bank).shape[0])
        ),
    ):
        bucket = int(row[0])
        if bucket % parts != rank:
            local_error = "A logical identity record reached the wrong bucket owner."
            continue
        entries.setdefault(bucket, []).append((bytes(row[1:33]), bytes(row[33:65])))
    bucket_digests = np.zeros((_OWNER_LOCAL_IDENTITY_BUCKET_COUNT, 33), dtype=np.uint8)
    for bucket in range(_OWNER_LOCAL_IDENTITY_BUCKET_COUNT):
        if bucket % parts != rank:
            continue
        records = sorted(entries.get(bucket, ()))
        if any(
            records[index - 1][0] == records[index][0] for index in range(1, len(records))
        ):
            local_error = (
                "Distributed scientific identity contains duplicate entity keys."
            )
            continue
        digest = canonical_fingerprint(
            {
                "kind": f"{kind}-logical-bucket",
                "bucket": bucket,
                "records": tuple(
                    (key_digest.hex(), record_digest.hex())
                    for key_digest, record_digest in records
                ),
            }
        )
        bucket_digests[bucket, 0] = 1
        bucket_digests[bucket, 1:] = np.frombuffer(bytes.fromhex(digest), dtype=np.uint8)
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_bucket",
        "distributed identity bucket validation",
    )
    gathered = _owner_group_allgather(
        bucket_digests, execution_group, f"{axis_name}_root"
    )
    presence = np.asarray(gathered[:, :, 0], dtype=np.int64)
    if np.any(np.sum(presence, axis=0) != 1):
        raise ValueError("Distributed identity logical buckets lack unique owners.")
    logical_buckets = tuple(
        (
            bucket,
            np.asarray(
                gathered[np.flatnonzero(presence[:, bucket])[0], bucket, 1:],
                dtype=np.uint8,
            )
            .tobytes()
            .hex(),
        )
        for bucket in range(_OWNER_LOCAL_IDENTITY_BUCKET_COUNT)
    )
    return canonical_fingerprint(
        {
            "kind": f"owner-local-{kind}",
            "source_id": source_id,
            "source_revision": source_revision,
            "header": header,
            "logical_buckets": logical_buckets,
        }
    )


def _owner_channel_exchange(
    plan: DistributedHaloPlan,
    local_values: np.ndarray,
    execution_group: Any,
    axis_name: str,
    /,
) -> np.ndarray:
    """Exchange one padded per-owner message bank through the shared halo route."""
    if not isinstance(plan, DistributedHaloPlan):
        raise TypeError("plan must be DistributedHaloPlan.")
    if local_values.ndim < 1 or local_values.shape[0] != plan.local_capacity:
        raise ValueError("Owner channel values must match the route's local capacity.")
    from ._overset_exchange import _owner_local_halo_exchange

    return _owner_local_halo_exchange(
        plan, local_values, execution_group.devices, axis_name
    )


def _owner_directory_bucket(identifiers: ArrayLike, part_count: int, /) -> np.ndarray:
    ids = np.asarray(identifiers)
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or part_count <= 0
        or np.any(ids < 0)
        or (
            ids.dtype.kind == "u"
            and ids.size
            and int(np.max(ids)) > np.iinfo(np.int64).max
        )
    ):
        raise ValueError("Owner directory IDs must be nonnegative int64 values.")
    values = ids.astype(np.uint64, copy=True)
    with np.errstate(over="ignore"):
        values ^= values >> np.uint64(30)
        values *= np.uint64(0xBF58476D1CE4E5B9)
        values ^= values >> np.uint64(27)
        values *= np.uint64(0x94D049BB133111EB)
        values ^= values >> np.uint64(31)
    return (values % np.uint64(part_count)).astype(np.int32)


def _owner_group_fail_if_any(
    local_error: str | None,
    execution_group: Any,
    axis_name: str,
    context: str,
    /,
) -> None:
    failed = np.asarray(
        _owner_group_allgather(
            np.asarray([local_error is not None], dtype=np.bool_),
            execution_group,
            axis_name,
        ),
        dtype=np.bool_,
    ).reshape(-1)
    if np.any(failed):
        raise ValueError(
            local_error
            if local_error is not None
            else f"Another owner failed {context} validation."
        )


_OWNER_REASON_BYTES = 768


def _owner_group_fail_with_reasons(
    local_error: str | None,
    execution_group: Any,
    axis_name: str,
    /,
) -> None:
    """Raise one refusal naming every failing owner's own reason on all owners."""
    payload = np.zeros((_OWNER_REASON_BYTES + 1,), dtype=np.uint8)
    if local_error is not None:
        raw = local_error.encode("utf-8")[:_OWNER_REASON_BYTES]
        payload[0] = 1
        payload[1 : 1 + len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    rows = np.asarray(
        _owner_group_allgather(payload, execution_group, axis_name), dtype=np.uint8
    ).reshape((-1, _OWNER_REASON_BYTES + 1))
    reasons = [
        f"owner {rank}: "
        + bytes(row[1:]).rstrip(b"\x00").decode("utf-8", errors="replace")
        for rank, row in enumerate(rows)
        if row[0]
    ]
    if reasons:
        raise ValueError("; ".join(reasons))


def _sharded_owner_directory_id(
    directory_kind: str,
    source_manifest_id: str,
    group_id: str,
    part_count: int,
    shard_records: tuple[tuple[int, str, int, str], ...],
    /,
) -> str:
    if len(shard_records) != part_count or part_count <= 0:
        raise ValueError("Owner directory evidence must contain one record per rank.")
    if any(
        len(record) != 4
        or record[0] < 0
        or record[2] < 0
        or len(bytes.fromhex(record[1])) != 32
        or len(bytes.fromhex(record[3])) != 32
        for record in shard_records
    ) or sum(record[0] for record in shard_records) != sum(
        record[2] for record in shard_records
    ):
        raise ValueError("Owner directory shard evidence is malformed.")
    return canonical_fingerprint(
        {
            "kind": "sharded-owner-directory",
            "directory_kind": directory_kind,
            "source_manifest": source_manifest_id,
            "group": group_id,
            "part_count": part_count,
            "entity_count": sum(record[0] for record in shard_records),
            "owner_count": sum(record[2] for record in shard_records),
            "shards": [list(record) for record in shard_records],
        }
    )


def _build_sharded_owner_directory(
    local_owned_ids: ArrayLike,
    rank: int,
    part_count: int,
    execution_group: Any,
    axis_name: str,
    source_manifest_id: str,
    directory_kind: str,
    /,
    *,
    owner_evidence: tuple[str | None, str | None] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    str,
    int,
    tuple[tuple[int, str, int, str], ...],
]:
    """Route owned IDs to stable hash shards; retain only this rank's bucket."""
    ids_raw = np.asarray(local_owned_ids)
    local_error = None
    try:
        if (
            ids_raw.ndim != 1
            or ids_raw.dtype.kind not in "iu"
            or np.any(ids_raw < 0)
            or (
                ids_raw.dtype.kind == "u"
                and ids_raw.size
                and int(np.max(ids_raw)) > np.iinfo(np.int64).max
            )
            or np.unique(ids_raw).size != ids_raw.size
            or not 0 <= rank < part_count
            or not source_manifest_id
            or not directory_kind
        ):
            raise ValueError(
                "Owned owner-directory IDs or source identity are malformed."
            )
    except Exception as error:
        local_error = str(error)
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "owner-directory input"
    )
    ids = ids_raw.astype(np.int64, copy=False)
    buckets = _owner_directory_bucket(ids, part_count)
    outgoing = []
    for destination in range(part_count):
        selected = (buckets == destination) & (destination != rank)
        values = np.empty((int(np.count_nonzero(selected)), 2), dtype=np.int64)
        values[:, 0] = ids[selected]
        values[:, 1] = rank
        outgoing.append(values)
    incoming = _owner_exchange_message_banks(
        tuple(outgoing), rank, execution_group, axis_name
    )
    local_bucket = buckets == rank
    shard_ids = [ids[local_bucket]]
    shard_owners = [np.full((int(np.count_nonzero(local_bucket)),), rank, dtype=np.int32)]
    local_error = None
    for sender, rows in incoming.items():
        if rows.ndim != 2 or rows.shape[1] != 2 or np.any(rows[:, 1] != sender):
            local_error = "Owner directory message has a mismatched sender rank."
            continue
        shard_ids.append(np.asarray(rows[:, 0], dtype=np.int64))
        shard_owners.append(np.asarray(rows[:, 1], dtype=np.int32))
    ids_shard = np.concatenate(shard_ids)
    owners_shard = np.concatenate(shard_owners)
    order = np.argsort(ids_shard, kind="stable")
    ids_shard, owners_shard = ids_shard[order], owners_shard[order]
    if np.unique(ids_shard).size != ids_shard.size:
        local_error = "The sharded owner directory contains a duplicate owned ID."
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "owner-directory duplicate"
    )
    if owner_evidence is not None and (
        len(owner_evidence) != 2
        or any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for value in owner_evidence
        )
    ):
        local_error = "Owner directory local evidence identities are malformed."
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "owner-directory evidence"
    )
    shard_digest = canonical_fingerprint(
        {
            "kind": directory_kind,
            "source_manifest": source_manifest_id,
            "rank": rank,
            "ids": ids_shard,
            "owners": owners_shard,
        }
    )
    owner_ids = np.sort(ids, kind="stable")
    owner_digest = canonical_fingerprint(
        {
            "kind": f"{directory_kind}-owner-bank",
            "source_manifest": source_manifest_id,
            "rank": rank,
            "ids": owner_ids,
            "owners": np.full((owner_ids.size,), rank, dtype=np.int32),
            "owner_evidence": owner_evidence,
        }
    )
    digest_bytes = np.frombuffer(bytes.fromhex(shard_digest), dtype=np.uint8)
    owner_digest_bytes = np.frombuffer(bytes.fromhex(owner_digest), dtype=np.uint8)
    local_record = np.concatenate(
        (
            np.asarray([ids_shard.size], dtype=np.int64).view(np.uint8),
            digest_bytes,
            np.asarray([owner_ids.size], dtype=np.int64).view(np.uint8),
            owner_digest_bytes,
        )
    )
    records = _owner_group_allgather(local_record, execution_group, axis_name)
    shard_counts = (
        np.asarray(records[:, :8], dtype=np.uint8).copy().view(np.int64).reshape(-1)
    )
    shard_digests = tuple(
        np.asarray(record[8:40], dtype=np.uint8).tobytes().hex() for record in records
    )
    owner_counts = (
        np.asarray(records[:, 40:48], dtype=np.uint8).copy().view(np.int64).reshape(-1)
    )
    owner_digests = tuple(
        np.asarray(record[48:80], dtype=np.uint8).tobytes().hex() for record in records
    )
    shard_records = tuple(
        (
            int(shard_counts[index]),
            shard_digests[index],
            int(owner_counts[index]),
            owner_digests[index],
        )
        for index in range(part_count)
    )
    directory_id = _sharded_owner_directory_id(
        directory_kind,
        source_manifest_id,
        execution_group.spec.group_id,
        part_count,
        shard_records,
    )
    return (
        ids_shard,
        owners_shard,
        directory_id,
        int(np.sum(shard_counts)),
        shard_records,
    )


def _owner_directory_lookup(
    query_ids: ArrayLike,
    shard_ids: np.ndarray,
    shard_owner_ranks: np.ndarray,
    local_rank: int,
    part_count: int,
    execution_group: Any,
    axis_name: str,
    directory_kind: str,
    /,
    *,
    expected_owner_ranks: ArrayLike | None = None,
) -> np.ndarray:
    """Lookup stable IDs through their hash-owner shards without an all-gather."""
    identifiers = np.asarray(query_ids)
    expected_owners = None
    local_error = None
    try:
        if (
            identifiers.ndim != 1
            or identifiers.dtype.kind not in "iu"
            or np.any(identifiers < 0)
            or (
                identifiers.dtype.kind == "u"
                and identifiers.size
                and int(np.max(identifiers)) > np.iinfo(np.int64).max
            )
            or not 0 <= local_rank < part_count
            or not directory_kind
        ):
            raise ValueError("Owner directory lookup IDs are malformed.")
        if expected_owner_ranks is not None:
            raw_expected_owners = np.asarray(expected_owner_ranks)
            if (
                raw_expected_owners.shape != identifiers.shape
                or raw_expected_owners.dtype.kind not in "iu"
                or np.any(raw_expected_owners < 0)
                or np.any(raw_expected_owners >= part_count)
            ):
                raise ValueError(
                    "Owner directory expected owner ranks do not match its local query bank."
                )
            expected_owners = raw_expected_owners.astype(np.int32, copy=False)
    except Exception as error:
        local_error = str(error)
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, f"{directory_kind} lookup input"
    )
    ids = identifiers.astype(np.int64, copy=False)
    buckets = _owner_directory_bucket(ids, part_count)
    outgoing_queries = []
    local_indices = np.flatnonzero(buckets == local_rank)
    local_owners = np.full((ids.size,), -1, dtype=np.int32)
    local_found = np.zeros((ids.size,), dtype=np.bool_)

    def locate(values: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
        positions = np.searchsorted(shard_ids, values)
        found = (positions < shard_ids.size) & (
            shard_ids[np.minimum(positions, max(shard_ids.size - 1, 0))] == values
            if shard_ids.size
            else False
        )
        owners = np.full(values.shape, -1, dtype=np.int32)
        if shard_ids.size:
            owners[found] = shard_owner_ranks[positions[found]]
        return owners, np.asarray(found, dtype=np.bool_)

    if local_indices.size:
        owners, found = locate(ids[local_indices])
        local_owners[local_indices] = owners
        local_found[local_indices] = found
    for destination in range(part_count):
        selected = np.flatnonzero((buckets == destination) & (destination != local_rank))
        bank = np.empty((selected.size, 2), dtype=np.int64)
        bank[:, 0] = selected
        bank[:, 1] = ids[selected]
        outgoing_queries.append(bank)
    incoming_queries = _owner_exchange_message_banks(
        tuple(outgoing_queries), local_rank, execution_group, axis_name
    )
    outgoing_replies = [np.zeros((0, 3), dtype=np.int64) for _ in range(part_count)]
    for requester, bank in incoming_queries.items():
        owners, found = locate(np.asarray(bank[:, 1], dtype=np.int64))
        reply = np.empty((bank.shape[0], 3), dtype=np.int64)
        reply[:, 0] = np.asarray(bank[:, 0], dtype=np.int64)
        reply[:, 1] = owners
        reply[:, 2] = found.astype(np.int64)
        outgoing_replies[requester] = reply
    incoming_replies = _owner_exchange_message_banks(
        tuple(outgoing_replies), local_rank, execution_group, axis_name
    )
    received = local_found.copy()
    reply_errors: list[str] = []
    for owner, bank in incoming_replies.items():
        response = np.asarray(bank, dtype=np.int64)
        if response.ndim != 2 or response.shape[1] != 3:
            reply_errors.append(
                f"{directory_kind} lookup from owner {owner} has malformed reply rows."
            )
            continue
        rows = response[:, 0]
        in_range = (rows >= 0) & (rows < ids.size)
        if np.any(~in_range):
            reply_errors.append(
                f"{directory_kind} lookup from owner {owner} returned out-of-range rows."
            )
        valid_positions = np.flatnonzero(in_range)
        if not valid_positions.size:
            continue
        valid_rows = rows[valid_positions]
        unique_rows, row_counts = np.unique(valid_rows, return_counts=True)
        repeated_rows = np.isin(valid_rows, unique_rows[row_counts > 1])
        previously_received = received[valid_rows]
        duplicate_rows = repeated_rows | previously_received
        if np.any(duplicate_rows):
            reply_errors.append(
                f"{directory_kind} lookup from owner {owner} returned duplicate rows."
            )
        statuses = response[valid_positions, 2]
        owners = response[valid_positions, 1]
        valid_status = (statuses == 0) | (statuses == 1)
        valid_owner = ((statuses == 0) & (owners == -1)) | (
            (statuses == 1) & (owners >= 0) & (owners < part_count)
        )
        if np.any(~valid_status) or np.any(~valid_owner):
            reply_errors.append(
                f"{directory_kind} lookup from owner {owner} returned invalid status/owner rows."
            )
        accepted = ~duplicate_rows & valid_status & valid_owner
        found = accepted & (statuses == 1)
        local_owners[valid_rows[found]] = owners[found].astype(np.int32, copy=False)
        received[valid_rows[found]] = True
    missing_rows = np.flatnonzero(~received)
    if expected_owners is None:
        missing_ids: object = tuple(int(ids[row]) for row in missing_rows)
    else:
        missing_ids = tuple(
            (int(ids[row]), int(expected_owners[row])) for row in missing_rows
        )
    local_error_parts = reply_errors
    if missing_rows.size:
        missing_label = "IDs" if expected_owners is None else "(ID, declared owner) rows"
        local_error_parts.append(
            f"{directory_kind} lookup left missing {missing_label} {missing_ids!r} "
            f"on rank {local_rank}."
        )
    local_error = "; ".join(local_error_parts) if local_error_parts else None
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, f"{directory_kind} lookup result"
    )
    return local_owners


@final
class ShardedOwnerDirectoryWitness(StrictModule, NonTrainableState):
    """Hash-sharded owner directories for one accepted local mesh result.

    Each rank stores only its local input/halo IDs and the hash buckets assigned
    to it. Global directory identity is a digest of the per-rank bucket digests;
    no rank retains a complete global cell/source/vertex owner table.
    """

    source_manifest_id: str = eqx.field(static=True)
    directory_entry_id: str = eqx.field(static=True)
    source_authority_id: str | None = eqx.field(static=True)
    local_source_map_id: str | None = eqx.field(static=True)
    local_source_context_id: str | None = eqx.field(static=True)
    group_id: str = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    source_directory_ids: Array
    source_directory_owner_ranks: Array
    cell_directory_ids: Array
    cell_directory_owner_ranks: Array
    vertex_directory_ids: Array
    vertex_directory_owner_ranks: Array
    local_source_point_ids: Array
    local_source_owner_ranks: Array
    local_source_owned: Array
    local_cell_ids: Array
    local_cell_owner_ranks: Array
    local_vertex_ids: Array
    local_vertex_owner_ranks: Array
    source_shard_records: tuple[tuple[int, str, int, str], ...] = eqx.field(static=True)
    cell_shard_records: tuple[tuple[int, str, int, str], ...] = eqx.field(static=True)
    vertex_shard_records: tuple[tuple[int, str, int, str], ...] = eqx.field(static=True)
    source_directory_id: str = eqx.field(static=True)
    cell_directory_id: str = eqx.field(static=True)
    vertex_directory_id: str = eqx.field(static=True)
    directory_id: str = eqx.field(static=True)
    authority_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        execution_group: ExecutionGroup,
        rank: int,
        source_manifest_id: str,
        directory_entry_id: str,
        source_point_ids: ArrayLike,
        source_owner_ranks: ArrayLike,
        source_owned: ArrayLike,
        cell_ids: ArrayLike,
        cell_owner_ranks: ArrayLike,
        vertex_ids: ArrayLike,
        vertex_owner_ranks: ArrayLike,
        /,
        *,
        source_authority_id: str | None = None,
        source_map_id: str | None = None,
        source_context_id: str | None = None,
        axis_name: str = "mesh_owner_directory",
        _validated_shards: ShardedOwnerDirectoryWitness | None = None,
    ) -> None:
        import jax

        if not isinstance(execution_group, ExecutionGroup):
            raise TypeError("execution_group must be ExecutionGroup.")
        members = execution_group.spec.process_indices
        devices = execution_group.devices
        if (
            len(devices) != len(members)
            or tuple(device.process_index for device in devices) != members
            or not execution_group.is_member
        ):
            raise ValueError(
                "Owner directories require one registered group device per process."
            )
        if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)):
            raise TypeError("rank must be an integer excluding bool.")
        rank_ = int(rank)
        parts = len(members)
        if (
            not 0 <= rank_ < parts
            or members[rank_] != jax.process_index()
            or not isinstance(source_manifest_id, str)
            or not source_manifest_id.strip()
            or not isinstance(directory_entry_id, str)
            or not directory_entry_id.strip()
            or not isinstance(axis_name, str)
            or not axis_name.strip()
            or source_authority_id is not None
            and (
                not isinstance(source_authority_id, str)
                or not source_authority_id.strip()
            )
            or source_map_id is not None
            and (not isinstance(source_map_id, str) or not source_map_id.strip())
            or source_context_id is not None
            and (not isinstance(source_context_id, str) or not source_context_id.strip())
        ):
            raise ValueError("Owner directory group or source identity is invalid.")
        manifest = source_manifest_id.strip()
        entry_id = directory_entry_id.strip()
        authority_id_value = (
            None if source_authority_id is None else source_authority_id.strip()
        )
        source_map_id_value = None if source_map_id is None else source_map_id.strip()
        source_context_id_value = (
            None if source_context_id is None else source_context_id.strip()
        )
        axis = axis_name.strip()
        if _validated_shards is not None:
            previous = _validated_shards
            local_error = None
            if type(previous) is not ShardedOwnerDirectoryWitness:
                local_error = "Authority rebinding requires a canonical owner witness."
            elif (
                previous.group_id != execution_group.spec.group_id
                or previous.rank != rank_
                or previous.part_count != parts
                or previous.source_manifest_id != manifest
                or previous.directory_entry_id != entry_id
                or previous.local_source_map_id != source_map_id_value
                or previous.local_source_context_id != source_context_id_value
                or (
                    previous.source_authority_id is not None
                    and previous.source_authority_id != authority_id_value
                )
            ):
                local_error = "Authority rebinding changed the accepted group, shard, or source binding."
            if local_error is None:
                try:
                    previous._require_stored()
                except Exception as error:
                    local_error = str(error)
            if local_error is None and (
                not np.array_equal(
                    np.asarray(source_point_ids, dtype=np.int64),
                    np.asarray(previous.local_source_point_ids, dtype=np.int64),
                )
                or not np.array_equal(
                    np.asarray(source_owner_ranks, dtype=np.int32),
                    np.asarray(previous.local_source_owner_ranks, dtype=np.int32),
                )
                or not np.array_equal(
                    np.asarray(source_owned, dtype=np.bool_),
                    np.asarray(previous.local_source_owned, dtype=np.bool_),
                )
                or not np.array_equal(
                    np.asarray(cell_ids, dtype=np.int64),
                    np.asarray(previous.local_cell_ids, dtype=np.int64),
                )
                or not np.array_equal(
                    np.asarray(cell_owner_ranks, dtype=np.int32),
                    np.asarray(previous.local_cell_owner_ranks, dtype=np.int32),
                )
                or not np.array_equal(
                    np.asarray(vertex_ids, dtype=np.int64),
                    np.asarray(previous.local_vertex_ids, dtype=np.int64),
                )
                or not np.array_equal(
                    np.asarray(vertex_owner_ranks, dtype=np.int32),
                    np.asarray(previous.local_vertex_owner_ranks, dtype=np.int32),
                )
            ):
                local_error = "Authority rebinding changed an accepted owner-local shard."
            _owner_group_fail_if_any(
                local_error, execution_group, axis, "owner-directory shard reuse"
            )
            authority_id = canonical_fingerprint(
                {
                    "kind": "sharded-owner-directory-authority",
                    "directory": previous.directory_id,
                    "source_authority": authority_id_value,
                }
            )
            authority_records = _owner_group_allgather(
                np.frombuffer(bytes.fromhex(authority_id), dtype=np.uint8),
                execution_group,
                axis,
            )
            local_error = (
                None
                if all(
                    np.asarray(record, dtype=np.uint8).tobytes().hex() == authority_id
                    for record in authority_records
                )
                else "Owner-directory ranks supplied different source authorities."
            )
            _owner_group_fail_if_any(
                local_error, execution_group, axis, "owner-directory authority identity"
            )
            self.source_manifest_id = previous.source_manifest_id
            self.directory_entry_id = previous.directory_entry_id
            self.source_authority_id = authority_id_value
            self.local_source_map_id = previous.local_source_map_id
            self.local_source_context_id = previous.local_source_context_id
            self.group_id = previous.group_id
            self.rank = previous.rank
            self.part_count = previous.part_count
            self.source_directory_ids = previous.source_directory_ids
            self.source_directory_owner_ranks = previous.source_directory_owner_ranks
            self.cell_directory_ids = previous.cell_directory_ids
            self.cell_directory_owner_ranks = previous.cell_directory_owner_ranks
            self.vertex_directory_ids = previous.vertex_directory_ids
            self.vertex_directory_owner_ranks = previous.vertex_directory_owner_ranks
            self.local_source_point_ids = previous.local_source_point_ids
            self.local_source_owner_ranks = previous.local_source_owner_ranks
            self.local_source_owned = previous.local_source_owned
            self.local_cell_ids = previous.local_cell_ids
            self.local_cell_owner_ranks = previous.local_cell_owner_ranks
            self.local_vertex_ids = previous.local_vertex_ids
            self.local_vertex_owner_ranks = previous.local_vertex_owner_ranks
            self.source_shard_records = previous.source_shard_records
            self.cell_shard_records = previous.cell_shard_records
            self.vertex_shard_records = previous.vertex_shard_records
            self.source_directory_id = previous.source_directory_id
            self.cell_directory_id = previous.cell_directory_id
            self.vertex_directory_id = previous.vertex_directory_id
            self.directory_id = previous.directory_id
            self.authority_id = authority_id
            return
        local_error = None

        def normalize_bank(
            name: str, identifiers: ArrayLike, owners: ArrayLike
        ) -> tuple[np.ndarray, np.ndarray]:
            raw_ids = np.asarray(identifiers)
            raw_owners = np.asarray(owners)
            if (
                raw_ids.ndim != 1
                or raw_ids.dtype.kind not in "iu"
                or raw_owners.shape != raw_ids.shape
                or raw_owners.dtype.kind not in "iu"
                or np.any(raw_ids < 0)
                or (
                    raw_ids.dtype.kind == "u"
                    and raw_ids.size
                    and int(np.max(raw_ids)) > np.iinfo(np.int64).max
                )
                or np.any(raw_owners < 0)
                or np.any(raw_owners >= parts)
                or np.unique(raw_ids).size != raw_ids.size
            ):
                raise ValueError(f"{name} IDs/owners are malformed or duplicated.")
            return (
                raw_ids.astype(np.int64, copy=False),
                raw_owners.astype(np.int32, copy=False),
            )

        try:
            source_ids, source_owners = normalize_bank(
                "Source point", source_point_ids, source_owner_ranks
            )
            source_owned = np.asarray(source_owned)
            if (
                source_owned.shape != source_ids.shape
                or source_owned.dtype != np.bool_
                or not np.array_equal(source_owned, source_owners == rank_)
            ):
                raise ValueError("Source point ownership flags disagree with rank.")
            cells, cell_owners = normalize_bank("Cell", cell_ids, cell_owner_ranks)
            vertices, vertex_owners = normalize_bank(
                "Vertex", vertex_ids, vertex_owner_ranks
            )
        except Exception as error:
            local_error = str(error)
            source_ids = np.zeros((0,), dtype=np.int64)
            source_owners = np.zeros((0,), dtype=np.int32)
            source_owned = np.zeros((0,), dtype=np.bool_)
            cells = np.zeros((0,), dtype=np.int64)
            cell_owners = np.zeros((0,), dtype=np.int32)
            vertices = np.zeros((0,), dtype=np.int64)
            vertex_owners = np.zeros((0,), dtype=np.int32)
        _owner_group_fail_if_any(
            local_error, execution_group, axis, "owner directory local input"
        )
        if local_error is not None:
            raise ValueError(local_error)

        manifest_record = np.frombuffer(
            bytes.fromhex(
                canonical_fingerprint(
                    {
                        "source_manifest": manifest,
                        "directory_entry": entry_id,
                        "group": execution_group.spec.group_id,
                        "parts": parts,
                    }
                )
            ),
            dtype=np.uint8,
        )
        manifest_records = _owner_group_allgather(manifest_record, execution_group, axis)
        if np.any(manifest_records != manifest_records[0]):
            raise ValueError(
                "Owner ranks supplied different source/directory identities."
            )

        source_shard = _build_sharded_owner_directory(
            source_ids[source_owned],
            rank_,
            parts,
            execution_group,
            axis,
            manifest,
            "source-points",
            owner_evidence=(source_map_id_value, source_context_id_value),
        )
        cell_shard = _build_sharded_owner_directory(
            cells[cell_owners == rank_],
            rank_,
            parts,
            execution_group,
            axis,
            manifest,
            "mesh-cells",
        )
        vertex_shard = _build_sharded_owner_directory(
            vertices[vertex_owners == rank_],
            rank_,
            parts,
            execution_group,
            axis,
            manifest,
            "mesh-vertices",
        )
        (
            source_directory_ids,
            source_directory_owners,
            source_directory_id,
            _,
            source_shard_records,
        ) = source_shard
        (
            cell_directory_ids,
            cell_directory_owners,
            cell_directory_id,
            _,
            cell_shard_records,
        ) = cell_shard
        (
            vertex_directory_ids,
            vertex_directory_owners,
            vertex_directory_id,
            _,
            vertex_shard_records,
        ) = vertex_shard
        for kind, local_ids, local_owners, directory_ids, directory_owners in (
            (
                "source-points",
                source_ids,
                source_owners,
                source_directory_ids,
                source_directory_owners,
            ),
            (
                "mesh-cells",
                cells,
                cell_owners,
                cell_directory_ids,
                cell_directory_owners,
            ),
            (
                "mesh-vertices",
                vertices,
                vertex_owners,
                vertex_directory_ids,
                vertex_directory_owners,
            ),
        ):
            resolved = _owner_directory_lookup(
                local_ids,
                directory_ids,
                directory_owners,
                rank_,
                parts,
                execution_group,
                axis,
                kind,
                expected_owner_ranks=(local_owners if kind == "mesh-vertices" else None),
            )
            local_error = (
                None
                if np.array_equal(resolved, local_owners)
                else f"{kind} owner ranks disagree with the sharded directory."
            )
            _owner_group_fail_if_any(
                local_error, execution_group, axis, f"{kind} owner lookup"
            )

        directory_id = canonical_fingerprint(
            {
                "kind": "sharded-mesh-owner-directory",
                "source_manifest": manifest,
                "group": execution_group.spec.group_id,
                "part_count": parts,
                "source": source_directory_id,
                "cells": cell_directory_id,
                "vertices": vertex_directory_id,
            }
        )
        authority_id = canonical_fingerprint(
            {
                "kind": "sharded-owner-directory-authority",
                "directory": directory_id,
                "source_authority": authority_id_value,
            }
        )
        self.source_manifest_id = manifest
        self.directory_entry_id = entry_id
        self.source_authority_id = authority_id_value
        self.local_source_map_id = source_map_id_value
        self.local_source_context_id = source_context_id_value
        self.group_id = execution_group.spec.group_id
        self.rank = rank_
        self.part_count = parts
        self.source_directory_ids = jnp.asarray(source_directory_ids, dtype=jnp.int64)
        self.source_directory_owner_ranks = jnp.asarray(
            source_directory_owners, dtype=jnp.int32
        )
        self.cell_directory_ids = jnp.asarray(cell_directory_ids, dtype=jnp.int64)
        self.cell_directory_owner_ranks = jnp.asarray(
            cell_directory_owners, dtype=jnp.int32
        )
        self.vertex_directory_ids = jnp.asarray(vertex_directory_ids, dtype=jnp.int64)
        self.vertex_directory_owner_ranks = jnp.asarray(
            vertex_directory_owners, dtype=jnp.int32
        )
        self.local_source_point_ids = jnp.asarray(source_ids, dtype=jnp.int64)
        self.local_source_owner_ranks = jnp.asarray(source_owners, dtype=jnp.int32)
        self.local_source_owned = jnp.asarray(source_owned, dtype=jnp.bool_)
        self.local_cell_ids = jnp.asarray(cells, dtype=jnp.int64)
        self.local_cell_owner_ranks = jnp.asarray(cell_owners, dtype=jnp.int32)
        self.local_vertex_ids = jnp.asarray(vertices, dtype=jnp.int64)
        self.local_vertex_owner_ranks = jnp.asarray(vertex_owners, dtype=jnp.int32)
        self.source_directory_id = source_directory_id
        self.cell_directory_id = cell_directory_id
        self.vertex_directory_id = vertex_directory_id
        self.source_shard_records = source_shard_records
        self.cell_shard_records = cell_shard_records
        self.vertex_shard_records = vertex_shard_records
        self.directory_id = directory_id
        self.authority_id = authority_id

    @classmethod
    def from_stored_fields(
        cls,
        value: ShardedOwnerDirectoryWitness,
        execution_group: ExecutionGroup,
        /,
        *,
        axis_name: str = "mesh_owner_directory",
    ) -> ShardedOwnerDirectoryWitness:
        import jax

        if not isinstance(execution_group, ExecutionGroup):
            raise TypeError("execution_group must be the current ExecutionGroup.")
        if not isinstance(axis_name, str) or not axis_name.strip():
            raise ValueError("axis_name must be a non-empty string.")
        members = execution_group.spec.process_indices
        devices = execution_group.devices
        local_error = None
        if not isinstance(value, cls):
            local_error = "Stored owner-directory value has the wrong type."
        else:
            try:
                value._require_stored()
            except Exception as error:
                local_error = str(error)
            if local_error is None and (
                len(devices) != len(members)
                or tuple(device.process_index for device in devices) != members
                or not execution_group.is_member
                or value.part_count != len(members)
                or value.group_id != execution_group.spec.group_id
                or value.rank < 0
                or value.rank >= len(members)
                or members[value.rank] != jax.process_index()
            ):
                local_error = (
                    "Stored owner-directory shard does not match the current group rank."
                )
        _owner_group_fail_if_any(
            local_error, execution_group, axis_name, "stored owner-directory input"
        )
        if not isinstance(value, cls):
            raise TypeError("Stored owner-directory value has the wrong type.")

        rank = value.rank
        parts = value.part_count
        directory_specs = (
            (
                "source-points",
                np.asarray(value.local_source_point_ids, dtype=np.int64)[
                    np.asarray(value.local_source_owned, dtype=np.bool_)
                ],
                value.local_source_point_ids,
                value.local_source_owner_ranks,
                value.source_directory_ids,
                value.source_directory_owner_ranks,
                value.source_directory_id,
                value.source_shard_records,
                (value.local_source_map_id, value.local_source_context_id),
            ),
            (
                "mesh-cells",
                np.asarray(value.local_cell_ids, dtype=np.int64)[
                    np.asarray(value.local_cell_owner_ranks, dtype=np.int32) == rank
                ],
                value.local_cell_ids,
                value.local_cell_owner_ranks,
                value.cell_directory_ids,
                value.cell_directory_owner_ranks,
                value.cell_directory_id,
                value.cell_shard_records,
                None,
            ),
            (
                "mesh-vertices",
                np.asarray(value.local_vertex_ids, dtype=np.int64)[
                    np.asarray(value.local_vertex_owner_ranks, dtype=np.int32) == rank
                ],
                value.local_vertex_ids,
                value.local_vertex_owner_ranks,
                value.vertex_directory_ids,
                value.vertex_directory_owner_ranks,
                value.vertex_directory_id,
                value.vertex_shard_records,
                None,
            ),
        )
        for (
            kind,
            local_owned_ids,
            local_ids,
            local_owners,
            stored_shard_ids,
            stored_shard_owners,
            stored_directory_id,
            stored_shard_records,
            owner_evidence,
        ) in directory_specs:
            rebuilt = _build_sharded_owner_directory(
                local_owned_ids,
                rank,
                parts,
                execution_group,
                axis_name,
                value.source_manifest_id,
                kind,
                owner_evidence=owner_evidence,
            )
            rebuilt_ids, rebuilt_owners, rebuilt_id, rebuilt_count, rebuilt_records = (
                rebuilt
            )
            local_error = None
            if (
                not np.array_equal(
                    rebuilt_ids, np.asarray(stored_shard_ids, dtype=np.int64)
                )
                or not np.array_equal(
                    rebuilt_owners, np.asarray(stored_shard_owners, dtype=np.int32)
                )
                or rebuilt_id != stored_directory_id
                or rebuilt_count != sum(record[0] for record in stored_shard_records)
                or rebuilt_records != stored_shard_records
            ):
                local_error = (
                    f"Restored {kind} owner shards differ from collective evidence."
                )
            _owner_group_fail_if_any(
                local_error,
                execution_group,
                axis_name,
                f"restored {kind} directory",
            )
            resolved_owners = _owner_directory_lookup(
                local_ids,
                rebuilt_ids,
                rebuilt_owners,
                rank,
                parts,
                execution_group,
                axis_name,
                kind,
                expected_owner_ranks=(local_owners if kind == "mesh-vertices" else None),
            )
            local_error = (
                None
                if np.array_equal(
                    resolved_owners, np.asarray(local_owners, dtype=np.int32)
                )
                else f"Restored {kind} owner ranks differ from the collective directory."
            )
            _owner_group_fail_if_any(
                local_error,
                execution_group,
                axis_name,
                f"restored {kind} owner lookup",
            )
        return value

    def require_collective_current(
        self,
        execution_group: Any,
        /,
        *,
        axis_name: str = "mesh_owner_directory",
    ) -> None:
        """Rebuild and compare all owner shards against the live collective bank."""
        self.from_stored_fields(self, execution_group, axis_name=axis_name)

    def _require_stored(self, /) -> None:
        if (
            not self.source_manifest_id
            or not self.directory_entry_id
            or not self.group_id
            or self.part_count <= 0
            or not 0 <= self.rank < self.part_count
            or self.source_authority_id is not None
            and (
                not isinstance(self.source_authority_id, str)
                or not self.source_authority_id
            )
            or self.local_source_map_id is not None
            and (
                not isinstance(self.local_source_map_id, str)
                or not self.local_source_map_id
            )
            or self.local_source_context_id is not None
            and (
                not isinstance(self.local_source_context_id, str)
                or not self.local_source_context_id
            )
        ):
            raise ValueError("Stored sharded owner-directory identity is malformed.")

        directory_specs = (
            (
                "source-points",
                self.source_directory_ids,
                self.source_directory_owner_ranks,
                self.source_directory_id,
                self.source_shard_records,
                self.local_source_point_ids,
                self.local_source_owner_ranks,
                self.local_source_owned,
                (self.local_source_map_id, self.local_source_context_id),
            ),
            (
                "mesh-cells",
                self.cell_directory_ids,
                self.cell_directory_owner_ranks,
                self.cell_directory_id,
                self.cell_shard_records,
                self.local_cell_ids,
                self.local_cell_owner_ranks,
                None,
                None,
            ),
            (
                "mesh-vertices",
                self.vertex_directory_ids,
                self.vertex_directory_owner_ranks,
                self.vertex_directory_id,
                self.vertex_shard_records,
                self.local_vertex_ids,
                self.local_vertex_owner_ranks,
                None,
                None,
            ),
        )
        for (
            kind,
            shard_ids_value,
            shard_owners_value,
            stored_directory_id,
            shard_records,
            local_ids_value,
            local_owners_value,
            local_owned_value,
            owner_evidence,
        ) in directory_specs:
            shard_ids = np.asarray(shard_ids_value)
            shard_owners = np.asarray(shard_owners_value)
            local_ids = np.asarray(local_ids_value)
            local_owners = np.asarray(local_owners_value)
            if (
                len(shard_records) != self.part_count
                or shard_ids.ndim != 1
                or shard_ids.dtype.kind not in "iu"
                or shard_owners.shape != shard_ids.shape
                or local_ids.ndim != 1
                or local_ids.dtype.kind not in "iu"
                or local_owners.shape != local_ids.shape
                or np.any(shard_ids < 0)
                or np.any(local_ids < 0)
                or np.any(shard_owners < 0)
                or np.any(shard_owners >= self.part_count)
                or np.any(local_owners < 0)
                or np.any(local_owners >= self.part_count)
                or np.unique(shard_ids).size != shard_ids.size
                or np.unique(local_ids).size != local_ids.size
                or np.any(shard_ids[1:] <= shard_ids[:-1])
                or np.any(
                    _owner_directory_bucket(shard_ids, self.part_count) != self.rank
                )
            ):
                raise ValueError("Stored sharded owner-directory arrays are malformed.")
            if local_owned_value is None:
                local_owned = local_owners == self.rank
            else:
                local_owned = np.asarray(local_owned_value)
                if (
                    local_owned.shape != local_ids.shape
                    or local_owned.dtype.kind != "b"
                    or np.any(local_owned != (local_owners == self.rank))
                ):
                    raise ValueError("Stored source ownership mask is inconsistent.")
            record = shard_records[self.rank]
            if len(record) != 4:
                raise ValueError("Stored owner-directory shard record is malformed.")
            bucket_count, bucket_digest, owner_count, owner_digest = record
            actual_bucket_digest = canonical_fingerprint(
                {
                    "kind": kind,
                    "source_manifest": self.source_manifest_id,
                    "rank": self.rank,
                    "ids": shard_ids,
                    "owners": shard_owners,
                }
            )
            owned_ids = np.sort(local_ids[local_owned], kind="stable")
            actual_owner_digest = canonical_fingerprint(
                {
                    "kind": f"{kind}-owner-bank",
                    "source_manifest": self.source_manifest_id,
                    "rank": self.rank,
                    "ids": owned_ids,
                    "owners": np.full((owned_ids.size,), self.rank, dtype=np.int32),
                    "owner_evidence": owner_evidence,
                }
            )
            if (
                bucket_count != shard_ids.size
                or bucket_digest != actual_bucket_digest
                or owner_count != owned_ids.size
                or owner_digest != actual_owner_digest
                or _sharded_owner_directory_id(
                    kind,
                    self.source_manifest_id,
                    self.group_id,
                    self.part_count,
                    shard_records,
                )
                != stored_directory_id
            ):
                raise ValueError("Stored sharded owner-directory evidence is stale.")

        expected_directory_id = canonical_fingerprint(
            {
                "kind": "sharded-mesh-owner-directory",
                "source_manifest": self.source_manifest_id,
                "group": self.group_id,
                "part_count": self.part_count,
                "source": self.source_directory_id,
                "cells": self.cell_directory_id,
                "vertices": self.vertex_directory_id,
            }
        )
        expected_authority_id = canonical_fingerprint(
            {
                "kind": "sharded-owner-directory-authority",
                "directory": expected_directory_id,
                "source_authority": self.source_authority_id,
            }
        )
        if (
            self.directory_id != expected_directory_id
            or self.authority_id != expected_authority_id
        ):
            raise ValueError("Stored sharded owner-directory identity is stale.")

    def bind_source_authority(
        self,
        source_authority_id: str,
        execution_group: ExecutionGroup,
        /,
        *,
        axis_name: str = "owner_directory_authority_binding",
    ) -> ShardedOwnerDirectoryWitness:
        identifier = str(source_authority_id).strip()
        return ShardedOwnerDirectoryWitness(
            execution_group,
            self.rank,
            self.source_manifest_id,
            self.directory_entry_id,
            self.local_source_point_ids,
            self.local_source_owner_ranks,
            self.local_source_owned,
            self.local_cell_ids,
            self.local_cell_owner_ranks,
            self.local_vertex_ids,
            self.local_vertex_owner_ranks,
            source_authority_id=identifier,
            source_map_id=self.local_source_map_id,
            source_context_id=self.local_source_context_id,
            axis_name=axis_name,
            _validated_shards=self,
        )

    def composition_entry(self, /) -> CompositionEntry:
        from ..lifecycle import CompositionEntry

        return CompositionEntry(
            self,
            entry_id=self.directory_entry_id,
            role="topology",
            owner_id="phydrax.meshing.distribution",
            structure_id=self.directory_id,
            revision_id=self.directory_id,
            semantics_id=self.source_manifest_id,
        )

    def _local_owner_ranks(
        self, identifiers: ArrayLike, directory_kind: str, /
    ) -> np.ndarray:
        match directory_kind:
            case "source-points":
                bank_ids, bank_owners = (
                    np.asarray(self.local_source_point_ids, dtype=np.int64),
                    np.asarray(self.local_source_owner_ranks, dtype=np.int32),
                )
            case "mesh-cells":
                bank_ids, bank_owners = (
                    np.asarray(self.local_cell_ids, dtype=np.int64),
                    np.asarray(self.local_cell_owner_ranks, dtype=np.int32),
                )
            case "mesh-vertices":
                bank_ids, bank_owners = (
                    np.asarray(self.local_vertex_ids, dtype=np.int64),
                    np.asarray(self.local_vertex_owner_ranks, dtype=np.int32),
                )
            case _:
                raise ValueError("Unknown local owner directory kind.")
        values = np.asarray(identifiers)
        if values.ndim != 1 or values.dtype.kind not in "iu" or np.any(values < 0):
            raise ValueError("Owner directory lookup IDs must be a nonnegative vector.")
        lookup = {
            int(identifier): int(owner)
            for identifier, owner in zip(bank_ids, bank_owners, strict=True)
        }
        try:
            return np.asarray(
                [lookup[int(identifier)] for identifier in values],
                dtype=np.int32,
            )
        except KeyError as error:
            raise ValueError(
                "Owner lookup ID is outside this rank's local source/mesh shard."
            ) from error

    def owner_ranks_for_ids(
        self, identifiers: ArrayLike, /, *, directory_kind: str = "source-points"
    ) -> np.ndarray:
        """Resolve IDs already present in this rank's validated local shard."""
        return self._local_owner_ranks(identifiers, directory_kind)

    def require_local_sources(
        self,
        source_authority_id: str,
        rank: int,
        point_ids: ArrayLike,
        owner_ranks: ArrayLike,
        owned: ArrayLike,
        /,
        *,
        point_windings: ArrayLike | None = None,
    ) -> None:
        identifiers = np.asarray(point_ids)
        owners = np.asarray(owner_ranks)
        owned_mask = np.asarray(owned)
        rank_ = operator.index(rank)
        if (
            self.source_authority_id is None
            or source_authority_id != self.source_authority_id
            or rank_ != self.rank
            or identifiers.ndim != 1
            or identifiers.dtype.kind not in "iu"
            or owners.shape != identifiers.shape
            or owners.dtype.kind not in "iu"
            or owned_mask.shape != identifiers.shape
            or owned_mask.dtype != np.bool_
            or not np.array_equal(owned_mask, owners == rank_)
        ):
            raise ValueError(
                "Local source rows do not match their accepted authority/rank."
            )
        expected_owners = self._local_owner_ranks(identifiers, "source-points")
        if not np.array_equal(expected_owners, owners):
            raise ValueError(
                "Local source rows disagree with the sharded owner directory."
            )
        if point_windings is None:
            if not np.array_equal(
                identifiers, np.asarray(self.local_source_point_ids, dtype=np.int64)
            ) or not np.array_equal(
                owners, np.asarray(self.local_source_owner_ranks, dtype=np.int32)
            ):
                raise ValueError("Local source bank differs from its prepared shard.")
            return
        windings = np.asarray(point_windings)
        if (
            windings.ndim != 2
            or windings.shape[0] != identifiers.size
            or windings.dtype.kind not in "iu"
            or len(
                {
                    (
                        int(identifier),
                        *(int(value) for value in winding),
                    )
                    for identifier, winding in zip(identifiers, windings, strict=True)
                }
            )
            != identifiers.size
        ):
            raise ValueError("Periodic source image rows need unique ID/winding keys.")

    def require_local_view(
        self,
        part_authority: CompositionEntry,
        accepted_composition: Composition,
        acceptance_receipt: CompositionRebindReceipt | None,
        part_id: str,
        source_mesh_id: str,
        source_topology_id: str,
        source_geometry_id: str,
        source_map_id: str | None,
        rank: int,
        cell_ids: ArrayLike,
        cell_owner_ranks: ArrayLike,
        owned: ArrayLike,
        vertex_ids: ArrayLike,
        vertex_owner_ranks: ArrayLike,
    ) -> None:
        from ..lifecycle import Composition, CompositionEntry, CompositionRebindReceipt

        rank_ = operator.index(rank)
        if (
            not isinstance(part_authority, CompositionEntry)
            or not isinstance(accepted_composition, Composition)
            or rank_ != self.rank
            or self.source_authority_id is None
            or part_authority.record_id != self.source_authority_id
            or accepted_composition.entry(part_authority.entry_id).record_id
            != part_authority.record_id
            or part_authority.revision_id != part_id
            or part_authority.structure_id != source_topology_id
            or part_authority.semantics_id != source_geometry_id
            or source_map_id != self.local_source_map_id
        ):
            raise ValueError(
                "Owner-local view does not match its accepted part authority."
            )
        if acceptance_receipt is not None:
            if not isinstance(acceptance_receipt, CompositionRebindReceipt) or (
                not acceptance_receipt.published
                or not acceptance_receipt.boundary_accepted
                or acceptance_receipt.composition.composition_id
                != accepted_composition.composition_id
            ):
                raise ValueError(
                    "Owner-local view is not at an accepted composition boundary."
                )
        directory_entry = accepted_composition.entry(self.directory_entry_id)
        if (
            not isinstance(directory_entry.value, ShardedOwnerDirectoryWitness)
            or directory_entry.value.directory_id != self.directory_id
            or directory_entry.structure_id != self.directory_id
            or directory_entry.value.authority_id != self.authority_id
            or not any(
                dependency.entry_id == self.directory_entry_id
                and dependency.facet == "structure"
                and dependency.bound_id == self.directory_id
                for dependency in part_authority.dependencies
            )
        ):
            raise ValueError(
                "Accepted part authority does not bind this owner directory."
            )
        if not any(
            dependency.facet == "structure"
            and accepted_composition.entry(dependency.entry_id).structure_id
            == source_mesh_id
            for dependency in part_authority.dependencies
        ):
            raise ValueError("Accepted part authority does not bind the source artifact.")
        cells = np.asarray(cell_ids, dtype=np.int64)
        cell_owners = np.asarray(cell_owner_ranks, dtype=np.int32)
        owned_mask = np.asarray(owned, dtype=np.bool_)
        vertices = np.asarray(vertex_ids, dtype=np.int64)
        vertex_owners = np.asarray(vertex_owner_ranks, dtype=np.int32)
        if (
            cells.ndim != 1
            or cell_owners.shape != cells.shape
            or owned_mask.shape != cells.shape
            or not np.array_equal(owned_mask, cell_owners == rank_)
            or not np.array_equal(cells, np.asarray(self.local_cell_ids))
            or not np.array_equal(cell_owners, np.asarray(self.local_cell_owner_ranks))
            or vertices.ndim != 1
            or vertex_owners.shape != vertices.shape
            or not np.array_equal(vertices, np.asarray(self.local_vertex_ids))
            or not np.array_equal(
                vertex_owners, np.asarray(self.local_vertex_owner_ranks)
            )
        ):
            raise ValueError(
                "Local cell/vertex ownership differs from the sharded directory."
            )


@final
class ShardedOwnerDirectoryData(StrictModule, NonTrainableState):
    """Scientific hash-sharded owner banks, without a lifecycle binding."""

    source_manifest_id: str = eqx.field(static=True)
    group_id: str = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    source_directory_ids: Array
    source_directory_owner_ranks: Array
    cell_directory_ids: Array
    cell_directory_owner_ranks: Array
    vertex_directory_ids: Array
    vertex_directory_owner_ranks: Array
    source_shard_records: tuple[tuple[int, str, int, str], ...] = eqx.field(static=True)
    cell_shard_records: tuple[tuple[int, str, int, str], ...] = eqx.field(static=True)
    vertex_shard_records: tuple[tuple[int, str, int, str], ...] = eqx.field(static=True)
    source_directory_id: str = eqx.field(static=True)
    cell_directory_id: str = eqx.field(static=True)
    vertex_directory_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_manifest_id: str,
        group_id: str,
        rank: int,
        part_count: int,
        source_directory_ids: ArrayLike,
        source_directory_owner_ranks: ArrayLike,
        cell_directory_ids: ArrayLike,
        cell_directory_owner_ranks: ArrayLike,
        vertex_directory_ids: ArrayLike,
        vertex_directory_owner_ranks: ArrayLike,
        source_shard_records: tuple[tuple[int, str, int, str], ...],
        cell_shard_records: tuple[tuple[int, str, int, str], ...],
        vertex_shard_records: tuple[tuple[int, str, int, str], ...],
        source_directory_id: str,
        cell_directory_id: str,
        vertex_directory_id: str,
        /,
    ) -> None:
        if (
            not isinstance(source_manifest_id, str)
            or not source_manifest_id
            or not isinstance(group_id, str)
            or not group_id
            or isinstance(rank, (bool, np.bool_))
            or not isinstance(rank, (int, np.integer))
            or isinstance(part_count, (bool, np.bool_))
            or not isinstance(part_count, (int, np.integer))
            or int(part_count) <= 0
            or not 0 <= int(rank) < int(part_count)
        ):
            raise ValueError(
                "Owner-directory data has invalid source/group/rank identity."
            )
        self.source_manifest_id = source_manifest_id
        self.group_id = group_id
        self.rank = int(rank)
        self.part_count = int(part_count)
        self.source_directory_ids = jnp.asarray(source_directory_ids, dtype=jnp.int64)
        self.source_directory_owner_ranks = jnp.asarray(
            source_directory_owner_ranks, dtype=jnp.int32
        )
        self.cell_directory_ids = jnp.asarray(cell_directory_ids, dtype=jnp.int64)
        self.cell_directory_owner_ranks = jnp.asarray(
            cell_directory_owner_ranks, dtype=jnp.int32
        )
        self.vertex_directory_ids = jnp.asarray(vertex_directory_ids, dtype=jnp.int64)
        self.vertex_directory_owner_ranks = jnp.asarray(
            vertex_directory_owner_ranks, dtype=jnp.int32
        )
        self.source_shard_records = tuple(source_shard_records)
        self.cell_shard_records = tuple(cell_shard_records)
        self.vertex_shard_records = tuple(vertex_shard_records)
        self.source_directory_id = source_directory_id
        self.cell_directory_id = cell_directory_id
        self.vertex_directory_id = vertex_directory_id
        self._require_stored()

    @classmethod
    def from_witness(
        cls, witness: ShardedOwnerDirectoryWitness, /
    ) -> ShardedOwnerDirectoryData:
        """Copy only scientific owner banks from a boundary witness."""
        if not isinstance(witness, ShardedOwnerDirectoryWitness):
            raise TypeError("witness must be ShardedOwnerDirectoryWitness.")
        witness._require_stored()
        return cls(
            witness.source_manifest_id,
            witness.group_id,
            witness.rank,
            witness.part_count,
            witness.source_directory_ids,
            witness.source_directory_owner_ranks,
            witness.cell_directory_ids,
            witness.cell_directory_owner_ranks,
            witness.vertex_directory_ids,
            witness.vertex_directory_owner_ranks,
            witness.source_shard_records,
            witness.cell_shard_records,
            witness.vertex_shard_records,
            witness.source_directory_id,
            witness.cell_directory_id,
            witness.vertex_directory_id,
        )

    def _validate_local_row_shapes(
        self,
        rank: int,
        distribution_id: str,
        source_map_id: str | None,
        source_context_id: str | None,
        source_point_ids: ArrayLike,
        source_owner_ranks: ArrayLike,
        source_owned: ArrayLike,
        cell_ids: ArrayLike,
        cell_owner_ranks: ArrayLike,
        owned: ArrayLike,
        vertex_ids: ArrayLike,
        vertex_owner_ranks: ArrayLike,
        /,
    ) -> None:
        if isinstance(rank, (bool, np.bool_)):
            raise TypeError("rank must be an integer excluding bool.")
        rank_ = operator.index(rank)
        expected_directory_id = canonical_fingerprint(
            {
                "kind": "sharded-mesh-owner-directory",
                "source_manifest": self.source_manifest_id,
                "group": self.group_id,
                "part_count": self.part_count,
                "source": self.source_directory_id,
                "cells": self.cell_directory_id,
                "vertices": self.vertex_directory_id,
            }
        )
        if rank_ != self.rank or distribution_id != expected_directory_id:
            raise ValueError("Local owner rows do not bind this directory identity.")
        source_ids = np.asarray(source_point_ids)
        source_owners = np.asarray(source_owner_ranks)
        source_mask = np.asarray(source_owned)
        cells = np.asarray(cell_ids)
        cell_owners = np.asarray(cell_owner_ranks)
        owned_mask = np.asarray(owned)
        vertices = np.asarray(vertex_ids)
        vertex_owners = np.asarray(vertex_owner_ranks)
        if (
            source_ids.ndim != 1
            or source_ids.dtype.kind not in "iu"
            or source_owners.shape != source_ids.shape
            or source_owners.dtype.kind not in "iu"
            or source_mask.shape != source_ids.shape
            or source_mask.dtype != np.bool_
            or not np.array_equal(source_mask, source_owners == rank_)
            or cells.ndim != 1
            or cells.dtype.kind not in "iu"
            or cell_owners.shape != cells.shape
            or cell_owners.dtype.kind not in "iu"
            or owned_mask.shape != cells.shape
            or owned_mask.dtype != np.bool_
            or not np.array_equal(owned_mask, cell_owners == rank_)
            or vertices.ndim != 1
            or vertices.dtype.kind not in "iu"
            or vertex_owners.shape != vertices.shape
            or vertex_owners.dtype.kind not in "iu"
            or source_map_id is not None
            and (not isinstance(source_map_id, str) or not source_map_id)
            or source_context_id is not None
            and (not isinstance(source_context_id, str) or not source_context_id)
        ):
            raise ValueError(
                "Local owner-directory rows or source-map identity are malformed."
            )

    def require_collective_current(
        self,
        execution_group: ExecutionGroup,
        rank: int,
        source_point_ids: ArrayLike,
        source_owner_ranks: ArrayLike,
        source_owned: ArrayLike,
        cell_ids: ArrayLike,
        cell_owner_ranks: ArrayLike,
        owned: ArrayLike,
        vertex_ids: ArrayLike,
        vertex_owner_ranks: ArrayLike,
        /,
        *,
        distribution_id: str,
        source_map_id: str | None,
        source_context_id: str | None,
        axis_name: str = "mesh_owner_directory",
    ) -> None:
        """Rebuild current owner shards without retaining their lifecycle issuer."""
        import jax

        if not isinstance(execution_group, ExecutionGroup):
            raise TypeError("execution_group must be the current ExecutionGroup.")
        if not isinstance(axis_name, str) or not axis_name.strip():
            raise ValueError("axis_name must be a non-empty string.")
        local_error = None
        try:
            self._require_stored()
            self._validate_local_row_shapes(
                rank,
                distribution_id,
                source_map_id,
                source_context_id,
                source_point_ids,
                source_owner_ranks,
                source_owned,
                cell_ids,
                cell_owner_ranks,
                owned,
                vertex_ids,
                vertex_owner_ranks,
            )
            members = execution_group.spec.process_indices
            devices = execution_group.devices
            if (
                len(devices) != len(members)
                or tuple(device.process_index for device in devices) != members
                or not execution_group.is_member
                or len(members) != self.part_count
                or execution_group.spec.group_id != self.group_id
                or self.rank != operator.index(rank)
                or self.rank < 0
                or members[self.rank] != jax.process_index()
            ):
                raise ValueError(
                    "Stored owner-directory data differs from the current group."
                )
        except Exception as error:
            local_error = str(error)
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            axis_name,
            "stored owner-directory data",
        )
        source_ids = np.asarray(source_point_ids, dtype=np.int64)
        source_owners = np.asarray(source_owner_ranks, dtype=np.int32)
        source_mask = np.asarray(source_owned, dtype=np.bool_)
        cells = np.asarray(cell_ids, dtype=np.int64)
        cell_owners = np.asarray(cell_owner_ranks, dtype=np.int32)
        owned_mask = np.asarray(owned, dtype=np.bool_)
        vertices = np.asarray(vertex_ids, dtype=np.int64)
        vertex_owners = np.asarray(vertex_owner_ranks, dtype=np.int32)
        directory_specs = (
            (
                "source-points",
                source_ids[source_mask],
                source_ids,
                source_owners,
                self.source_directory_ids,
                self.source_directory_owner_ranks,
                self.source_directory_id,
                self.source_shard_records,
                (source_map_id, source_context_id),
            ),
            (
                "mesh-cells",
                cells[owned_mask],
                cells,
                cell_owners,
                self.cell_directory_ids,
                self.cell_directory_owner_ranks,
                self.cell_directory_id,
                self.cell_shard_records,
                None,
            ),
            (
                "mesh-vertices",
                vertices[vertex_owners == self.rank],
                vertices,
                vertex_owners,
                self.vertex_directory_ids,
                self.vertex_directory_owner_ranks,
                self.vertex_directory_id,
                self.vertex_shard_records,
                None,
            ),
        )
        for (
            kind,
            local_owned_ids,
            local_ids,
            local_owners,
            stored_ids,
            stored_owners,
            stored_directory_id,
            stored_records,
            owner_evidence,
        ) in directory_specs:
            (
                rebuilt_ids,
                rebuilt_owners,
                rebuilt_directory_id,
                rebuilt_count,
                rebuilt_records,
            ) = _build_sharded_owner_directory(
                local_owned_ids,
                self.rank,
                self.part_count,
                execution_group,
                axis_name,
                self.source_manifest_id,
                kind,
                owner_evidence=owner_evidence,
            )
            local_error = None
            if (
                not np.array_equal(rebuilt_ids, np.asarray(stored_ids, dtype=np.int64))
                or not np.array_equal(
                    rebuilt_owners, np.asarray(stored_owners, dtype=np.int32)
                )
                or rebuilt_directory_id != stored_directory_id
                or rebuilt_count != sum(record[0] for record in stored_records)
                or rebuilt_records != stored_records
            ):
                local_error = (
                    f"Restored {kind} owner shards differ from collective evidence."
                )
            _owner_group_fail_if_any(
                local_error,
                execution_group,
                axis_name,
                f"restored {kind} directory",
            )
            resolved_owners = _owner_directory_lookup(
                local_ids,
                rebuilt_ids,
                rebuilt_owners,
                self.rank,
                self.part_count,
                execution_group,
                axis_name,
                kind,
                expected_owner_ranks=(local_owners if kind == "mesh-vertices" else None),
            )
            local_error = (
                None
                if np.array_equal(
                    resolved_owners, np.asarray(local_owners, dtype=np.int32)
                )
                else f"Restored {kind} owner ranks differ from the collective directory."
            )
            _owner_group_fail_if_any(
                local_error,
                execution_group,
                axis_name,
                f"restored {kind} owner lookup",
            )

    def _require_stored(self, /) -> None:
        if (
            not self.source_manifest_id
            or not self.group_id
            or self.part_count <= 0
            or not 0 <= self.rank < self.part_count
        ):
            raise ValueError("Stored owner-directory data identity is malformed.")
        for (
            kind,
            ids_value,
            owners_value,
            directory_id,
            records,
        ) in (
            (
                "source-points",
                self.source_directory_ids,
                self.source_directory_owner_ranks,
                self.source_directory_id,
                self.source_shard_records,
            ),
            (
                "mesh-cells",
                self.cell_directory_ids,
                self.cell_directory_owner_ranks,
                self.cell_directory_id,
                self.cell_shard_records,
            ),
            (
                "mesh-vertices",
                self.vertex_directory_ids,
                self.vertex_directory_owner_ranks,
                self.vertex_directory_id,
                self.vertex_shard_records,
            ),
        ):
            ids = np.asarray(ids_value)
            owners = np.asarray(owners_value)
            if (
                ids.ndim != 1
                or ids.dtype.kind not in "iu"
                or owners.shape != ids.shape
                or owners.dtype.kind not in "iu"
                or np.any(ids < 0)
                or np.any(owners < 0)
                or np.any(owners >= self.part_count)
                or np.unique(ids).size != ids.size
                or np.any(ids[1:] <= ids[:-1])
                or np.any(_owner_directory_bucket(ids, self.part_count) != self.rank)
                or len(records) != self.part_count
                or any(
                    len(record) != 4
                    or record[0] < 0
                    or not isinstance(record[1], str)
                    or not record[1]
                    or record[2] < 0
                    or not isinstance(record[3], str)
                    or not record[3]
                    for record in records
                )
                or records[self.rank][0] != ids.size
                or records[self.rank][1]
                != canonical_fingerprint(
                    {
                        "kind": kind,
                        "source_manifest": self.source_manifest_id,
                        "rank": self.rank,
                        "ids": ids,
                        "owners": owners,
                    }
                )
                or _sharded_owner_directory_id(
                    kind,
                    self.source_manifest_id,
                    self.group_id,
                    self.part_count,
                    records,
                )
                != directory_id
            ):
                raise ValueError("Stored owner-directory shard data is malformed.")
        if (
            not self.source_directory_id
            or not self.cell_directory_id
            or not self.vertex_directory_id
        ):
            raise ValueError("Stored owner-directory component identity is malformed.")


def _stored_owner_directory_banks(
    stored_shards: Sequence[ShardedOwnerDirectoryData],
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> tuple[str, int, tuple[np.ndarray, ...]]:
    local_error = None
    expected_ids: tuple[np.ndarray, ...] = ()
    source_manifest_id = ""
    old_part_count = 0
    try:
        shards = tuple(stored_shards)
        if not shards or any(
            not isinstance(shard, ShardedOwnerDirectoryData) for shard in shards
        ):
            raise TypeError(
                "stored_shards must contain ShardedOwnerDirectoryData records."
            )
        first = shards[0]
        source_manifest_id = first.source_manifest_id
        old_group_id = first.group_id
        old_part_count = first.part_count
        if (
            len(shards) != old_part_count
            or not source_manifest_id
            or not old_group_id
            or len({shard.rank for shard in shards}) != old_part_count
            or tuple(sorted(shard.rank for shard in shards))
            != tuple(range(old_part_count))
            or any(
                shard.source_manifest_id != source_manifest_id
                or shard.group_id != old_group_id
                or shard.part_count != old_part_count
                or shard.source_directory_id != first.source_directory_id
                or shard.cell_directory_id != first.cell_directory_id
                or shard.vertex_directory_id != first.vertex_directory_id
                or shard.source_shard_records != first.source_shard_records
                or shard.cell_shard_records != first.cell_shard_records
                or shard.vertex_shard_records != first.vertex_shard_records
                for shard in shards
            )
        ):
            raise ValueError(
                "Saved owner-directory shards do not form one complete placement."
            )
        for shard in shards:
            shard._require_stored()
        component_ids = (
            tuple(
                np.asarray(shard.source_directory_ids, dtype=np.int64) for shard in shards
            ),
            tuple(
                np.asarray(shard.cell_directory_ids, dtype=np.int64) for shard in shards
            ),
            tuple(
                np.asarray(shard.vertex_directory_ids, dtype=np.int64) for shard in shards
            ),
        )
        component_owners = (
            tuple(
                np.asarray(shard.source_directory_owner_ranks, dtype=np.int32)
                for shard in shards
            ),
            tuple(
                np.asarray(shard.cell_directory_owner_ranks, dtype=np.int32)
                for shard in shards
            ),
            tuple(
                np.asarray(shard.vertex_directory_owner_ranks, dtype=np.int32)
                for shard in shards
            ),
        )
        global_ids: list[np.ndarray] = []
        for name, ids_by_rank, owners_by_rank in zip(
            ("source", "cells", "vertices"),
            component_ids,
            component_owners,
            strict=True,
        ):
            ids = np.concatenate(ids_by_rank)
            owners = np.concatenate(owners_by_rank)
            if (
                ids.ndim != 1
                or owners.shape != ids.shape
                or np.any(ids < 0)
                or np.any(owners < 0)
                or np.any(owners >= old_part_count)
                or np.unique(ids).size != ids.size
            ):
                raise ValueError(
                    f"Saved {name} owner-directory bank is incomplete or duplicated."
                )
            global_ids.append(np.sort(ids, kind="stable"))
        expected_ids = tuple(global_ids)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_stored",
        "stored owner-directory repack input",
    )
    payload = canonical_fingerprint(
        {
            "source_manifest": source_manifest_id,
            "saved_part_count": old_part_count,
            "components": tuple(
                canonical_array_hex_fingerprint(values) for values in expected_ids
            ),
        }
    )
    gathered = _owner_group_allgather(
        np.frombuffer(bytes.fromhex(payload), dtype=np.uint8),
        execution_group,
        f"{axis_name}_stored_identity",
    )
    identities = tuple(
        np.asarray(row, dtype=np.uint8).tobytes().hex() for row in gathered
    )
    local_error = (
        None
        if len(set(identities)) == 1
        else "Saved owner shards differ across restore processes."
    )
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_stored_identity",
        "stored owner-directory agreement",
    )
    return source_manifest_id, old_part_count, expected_ids


def repack_sharded_owner_directory(
    stored_shards: Sequence[ShardedOwnerDirectoryData],
    execution_group: ExecutionGroup,
    local_source_point_ids: ArrayLike,
    local_source_owner_ranks: ArrayLike,
    local_source_owned: ArrayLike,
    local_cell_ids: ArrayLike,
    local_cell_owner_ranks: ArrayLike,
    local_vertex_ids: ArrayLike,
    local_vertex_owner_ranks: ArrayLike,
    /,
    *,
    directory_entry_id: str,
    source_map_id: str | None = None,
    source_context_id: str | None = None,
    axis_name: str = "restart_owner_directory_repack",
) -> ShardedOwnerDirectoryWitness:
    """Rehash complete archived directory banks for the current group placement.

    The full set of old shard artifacts is an explicit input. A single saved
    shard or its digest is insufficient to restore a different owner count.
    Current local ownership rows determine the new directory owner map.
    """
    import jax

    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    if not isinstance(axis_name, str) or not axis_name.strip():
        raise ValueError("axis_name must be a non-empty string.")
    if not isinstance(directory_entry_id, str) or not directory_entry_id.strip():
        raise ValueError("directory_entry_id must be a non-empty string.")
    axis = axis_name.strip()
    members = execution_group.spec.process_indices
    try:
        rank = members.index(jax.process_index())
    except ValueError as error:
        raise ValueError("The process is outside the restart owner group.") from error
    part_count = len(members)
    source_manifest_id, _, expected_ids = _stored_owner_directory_banks(
        stored_shards, execution_group, axis
    )
    witness = ShardedOwnerDirectoryWitness(
        execution_group,
        rank,
        source_manifest_id,
        directory_entry_id,
        local_source_point_ids,
        local_source_owner_ranks,
        local_source_owned,
        local_cell_ids,
        local_cell_owner_ranks,
        local_vertex_ids,
        local_vertex_owner_ranks,
        source_map_id=source_map_id,
        source_context_id=source_context_id,
        axis_name=f"{axis}_rebuild",
    )
    component_specs = (
        (
            "source-points",
            expected_ids[0],
            np.asarray(witness.source_directory_ids, dtype=np.int64),
            np.asarray(witness.source_directory_owner_ranks, dtype=np.int32),
            witness.source_shard_records,
        ),
        (
            "mesh-cells",
            expected_ids[1],
            np.asarray(witness.cell_directory_ids, dtype=np.int64),
            np.asarray(witness.cell_directory_owner_ranks, dtype=np.int32),
            witness.cell_shard_records,
        ),
        (
            "mesh-vertices",
            expected_ids[2],
            np.asarray(witness.vertex_directory_ids, dtype=np.int64),
            np.asarray(witness.vertex_directory_owner_ranks, dtype=np.int32),
            witness.vertex_shard_records,
        ),
    )
    for kind, old_ids, shard_ids, shard_owners, shard_records in component_specs:
        owner_count = sum(record[2] for record in shard_records)
        local_error = (
            None
            if owner_count == old_ids.size
            else f"Repacked {kind} owner directory changed its stable-ID count."
        )
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            f"{axis}_{kind}_count",
            "restart owner directory count",
        )
        queries = old_ids[rank::part_count]
        resolved = _owner_directory_lookup(
            queries,
            shard_ids,
            shard_owners,
            rank,
            part_count,
            execution_group,
            f"{axis}_{kind}_coverage",
            kind,
        )
        local_error = (
            None
            if np.all(resolved >= 0)
            else f"Repacked {kind} owner directory omits a saved stable ID."
        )
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            f"{axis}_{kind}_coverage",
            "restart owner directory coverage",
        )
    witness._require_stored()
    return witness


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
            layout = _periodic_cell_layout(carrier.mesh)
            return (
                mesh_cell_adjacency(carrier.mesh)
                if layout is None
                else _periodic_lifted_adjacency(carrier.mesh, layout)
            )
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
        partition_feasible: bool = True,
    ) -> None:
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
        periodic_mesh, periodic_layout = _periodic_layout(part)
        weights = _cell_weights(cell_weights, native.size)[position]
        evidence_owner = owner
        evidence_weights = weights
        evidence_adjacency = None
        if periodic_layout is not None:
            periodic_mesh = _require_periodic_mesh(periodic_mesh)
            evidence_owner = _periodic_collapse_owners(owner, periodic_layout)
            evidence_weights, weights = _periodic_quotient_weights(
                weights, periodic_layout
            )
            evidence_adjacency = _periodic_quotient_adjacency(periodic_mesh)
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
        if periodic_layout is not None:
            periodic_mesh = _require_periodic_mesh(periodic_mesh)
            adjacency = _periodic_lifted_adjacency(periodic_mesh, periodic_layout)
        else:
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
            evidence_owner,
            evidence_weights,
            adjacency if evidence_adjacency is None else evidence_adjacency,
            halo_rows.size,
            partition.part_count,
            partition_kind,
            str(partition_provenance),
            partition_feasible,
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

    def require_current(self, part: MeshPart, /) -> None:
        if not isinstance(part, MeshPart):
            raise TypeError("require_current requires a MeshPart.")
        if part.part_id != self.part.part_id:
            raise ValueError("Mesh distribution is stale for this part revision.")

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

    def local_cell_rows(self, rank: int, /, *, include_halo: bool = True) -> Array:
        """Return canonical cell rows held by one rank, owned rows then halo rows."""
        rank_ = operator.index(rank)
        if not 0 <= rank_ < self.partition.part_count:
            raise ValueError("Distribution rank is out of range.")
        owned = self.owned_rows[rank_]
        return jnp.concatenate((owned, self.halo_rows[rank_])) if include_halo else owned

    def local_cell_ids(self, rank: int, /, *, include_halo: bool = True) -> Array:
        """Stable canonical IDs corresponding to :meth:`local_cell_rows`."""
        return self.cell_global_ids[self.local_cell_rows(rank, include_halo=include_halo)]

    def expand_halos(
        self, requested_global_ids: tuple[ArrayLike, ...], /
    ) -> MeshDistribution:
        """Add discovered remote cells and preserve periodic orbit residency.

        Each rank supplies only the IDs its local search discovered. Callers
        exchange/expand repeatedly until their search reports no unresolved
        off-owner candidates; a configured ``halo_width`` is never a search
        completeness claim.
        """
        requests = tuple(np.asarray(value) for value in requested_global_ids)
        part_count = self.partition.part_count
        if len(requests) != part_count or any(
            value.ndim != 1 or (value.size and value.dtype.kind not in "iu")
            for value in requests
        ):
            raise ValueError("One integer request vector is required per owner rank.")
        native = np.asarray(self.cell_global_ids, dtype=np.int64)
        owner = np.asarray(self.partition.cell_owner, dtype=np.int64)
        periodic_mesh, periodic_layout = _periodic_layout(self.part)
        halo_ids = []
        changed = False
        for rank, request in enumerate(requests):
            ids = request.astype(np.int64, copy=False)
            if periodic_layout is not None:
                periodic_mesh = _require_periodic_mesh(periodic_mesh)
                closure = periodic_orbit_closure(
                    periodic_mesh, periodic_mesh.topological_dimension, ids
                )
                ids = closure.require_source(
                    periodic_mesh, periodic_mesh.topological_dimension
                )
            additions = np.unique(
                np.concatenate(
                    (np.asarray(self.halo_global_ids[rank], dtype=np.int64), ids)
                )
            )
            rows = _rows_of(native, additions)
            if np.any(rows < 0):
                raise ValueError(
                    "Halo expansion references a cell outside this mesh revision."
                )
            additions = additions[owner[rows] != rank]
            previous = np.asarray(self.halo_global_ids[rank], dtype=np.int64)
            changed = changed or not np.array_equal(additions, previous)
            halo_ids.append(additions)
        if not changed:
            return self
        weights = np.asarray(self.cell_weights, dtype=np.float64)
        if periodic_layout is not None:
            orbit_counts = np.bincount(
                periodic_layout.orbit_rows, minlength=periodic_layout.entity_ids.size
            )
            weights = weights * orbit_counts[periodic_layout.orbit_rows]
        return MeshDistribution(
            self.part,
            self.partition,
            cell_global_ids=native,
            halo_global_ids=tuple(halo_ids),
            split_factors=self.split_factors,
            halo_width=self.halo_width,
            cell_weights=weights,
            partition_kind=self.evidence.kind,
            partition_provenance=self.evidence.provenance,
            partition_feasible=self.evidence.feasible,
        )

    def local_cell_geometry(
        self, rank: int, /, *, include_halo: bool = True
    ) -> _LocalCellGeometry:
        """Select compact local cell/coordinate arrays from the canonical result.

        The returned record is an owner-local data view, not a second mesh or
        an acceptance result. Cell and coordinate row maps retain the source
        canonical ordering; cell IDs are sorted for deterministic conflict work.
        """
        if not isinstance(self.part.carrier, CellMeshingResult):
            raise TypeError(
                "Owner-local geometry selection requires a CellMeshingResult carrier."
            )
        result = self.part.carrier
        mesh = result.mesh
        rows = np.asarray(
            self.local_cell_rows(rank, include_halo=include_halo), dtype=np.int64
        )
        ids = np.asarray(self.cell_global_ids, dtype=np.int64)[rows]
        owned = np.isin(
            rows, np.asarray(self.owned_rows[operator.index(rank)], dtype=np.int64)
        )
        order = np.argsort(ids, kind="stable")
        rows, ids, owned = rows[order], ids[order], owned[order]
        elements, coordinate_routes, coordinates = result.geometry.resolve(mesh)
        block_counts = np.asarray(
            [block.cell_count for block in mesh.blocks], dtype=np.int64
        )
        block_offsets = np.concatenate(
            (np.zeros((1,), dtype=np.int64), np.cumsum(block_counts))
        )
        selected_records = []
        coordinate_rows: list[np.ndarray] = []
        vertex_rows: list[np.ndarray] = []
        for block_index, (block, element, route) in enumerate(
            zip(mesh.blocks, elements, coordinate_routes, strict=True)
        ):
            start, stop = map(int, block_offsets[block_index : block_index + 2])
            selected = np.flatnonzero((rows >= start) & (rows < stop))
            local_rows = rows[selected] - start
            cell_vertices = np.asarray(block.vertices, dtype=np.int64)[local_rows]
            vertex_valid = np.asarray(block.vertex_valid, dtype=np.bool_)[local_rows]
            dof_routes = np.asarray(route, dtype=np.int64)[local_rows]
            coordinate_rows.append(dof_routes[dof_routes >= 0])
            vertex_rows.append(cell_vertices[vertex_valid])
            selected_records.append(
                (
                    block,
                    element,
                    selected,
                    local_rows,
                    cell_vertices,
                    vertex_valid,
                    dof_routes,
                )
            )
        unique_coordinates = (
            np.unique(np.concatenate(coordinate_rows))
            if any(values.size for values in coordinate_rows)
            else np.zeros((0,), dtype=np.int64)
        )
        unique_vertices = (
            np.unique(np.concatenate(vertex_rows))
            if any(values.size for values in vertex_rows)
            else np.zeros((0,), dtype=np.int64)
        )
        local_blocks: list[_LocalCellGeometryBlock] = []
        for (
            block,
            element,
            selected,
            local_rows,
            cell_vertices,
            vertex_valid,
            dof_routes,
        ) in selected_records:
            local_vertices = np.where(
                vertex_valid,
                np.searchsorted(unique_vertices, np.maximum(cell_vertices, 0)),
                -1,
            ).astype(np.int32)
            local_coordinate_routes = np.where(
                dof_routes >= 0,
                np.searchsorted(unique_coordinates, np.maximum(dof_routes, 0)),
                -1,
            ).astype(np.int32)
            local_blocks.append(
                {
                    "name": block.name,
                    "cell_kind": block.cell_kind,
                    "cell_rows": local_rows,
                    "cell_ids": ids[selected],
                    "vertices": local_vertices,
                    "vertex_valid": vertex_valid,
                    "coordinate_routes": local_coordinate_routes,
                    "element": element,
                }
            )
        return {
            "part_id": self.part.part_id,
            "distribution_id": self.distribution_id,
            "cell_rows": rows,
            "cell_ids": ids,
            "owned": owned,
            "vertex_rows": unique_vertices,
            "vertex_ids": np.asarray(mesh.vertex_global_ids, dtype=np.int64)[
                unique_vertices
            ],
            "vertex_coordinates": np.asarray(mesh.coordinates)[unique_vertices],
            "coordinate_rows": unique_coordinates,
            "coordinates": np.asarray(coordinates)[unique_coordinates],
            "geometry_layout_id": result.geometry.geometry_layout_id,
            "blocks": tuple(local_blocks),
        }

    def owner_local_geometry_record(
        self,
        rank: int,
        part_authority: CompositionEntry,
        accepted_composition: Composition,
        owner_directory_witness: ShardedOwnerDirectoryWitness,
        /,
        *,
        source_id: str | None = None,
        source_revision: str | None = None,
        acceptance_receipt: CompositionRebindReceipt | None = None,
        include_halo: bool = True,
    ) -> dict[str, Any]:
        """Bind a serial distribution slice to the accepted owner-local view contract.

        This adapter is for serial inputs that already carry one complete
        ``MeshPart``. Owner-local native generation supplies local mesh shards
        directly and never calls this whole-result conversion.
        """
        from ..lifecycle import Composition, CompositionEntry, CompositionRebindReceipt

        if not isinstance(part_authority, CompositionEntry) or not isinstance(
            accepted_composition, Composition
        ):
            raise TypeError(
                "Owner-local source binding requires a CompositionEntry in its accepted Composition."
            )
        if acceptance_receipt is not None and not isinstance(
            acceptance_receipt, CompositionRebindReceipt
        ):
            raise TypeError(
                "acceptance_receipt must be CompositionRebindReceipt or None."
            )
        if acceptance_receipt is not None and (
            not acceptance_receipt.published
            or not acceptance_receipt.boundary_accepted
            or acceptance_receipt.composition.composition_id
            != accepted_composition.composition_id
        ):
            raise ValueError("Owner-local source binding is not an accepted boundary.")
        if (
            accepted_composition.entry(part_authority.entry_id).record_id
            != part_authority.record_id
        ):
            raise ValueError("Part authority is not the accepted composition entry.")
        if not isinstance(self.part.carrier, CellMeshingResult):
            raise TypeError(
                "Owner-local geometry conversion requires a CellMeshingResult carrier."
            )
        if not isinstance(include_halo, bool):
            raise TypeError("include_halo must be a bool.")
        rank_ = operator.index(rank)
        if not 0 <= rank_ < self.partition.part_count:
            raise ValueError("Distribution rank is out of range.")
        if (
            not isinstance(owner_directory_witness, ShardedOwnerDirectoryWitness)
            or owner_directory_witness.rank != rank_
            or owner_directory_witness.source_authority_id != part_authority.record_id
        ):
            raise ValueError(
                "Owner directory witness is not bound to this accepted part."
            )
        directory_entry = accepted_composition.entry(
            owner_directory_witness.directory_entry_id
        )
        if (
            directory_entry.value is not owner_directory_witness
            or directory_entry.structure_id != owner_directory_witness.directory_id
            or not any(
                dependency.entry_id == owner_directory_witness.directory_entry_id
                and dependency.facet == "structure"
                and dependency.bound_id == owner_directory_witness.directory_id
                for dependency in part_authority.dependencies
            )
        ):
            raise ValueError(
                "Accepted part authority does not bind its directory witness."
            )
        for name, value in (
            ("source_id", source_id),
            ("source_revision", source_revision),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string when supplied.")
        source_id = None if source_id is None else source_id.strip()
        source_revision = None if source_revision is None else source_revision.strip()

        from .providers._native_periodic import NativePeriodicSource

        authority_value = part_authority.value
        result = self.part.carrier
        mesh = result.mesh
        geometry = result.geometry
        source_map = geometry.source_map
        source_map_id = geometry.source_map_id
        if source_map is None:
            if isinstance(authority_value, NativePeriodicSource):
                map_source_id = authority_value.source_id
                map_source_revision = authority_value.source_revision
                if (source_id is not None and source_id != map_source_id) or (
                    source_revision is not None and source_revision != map_source_revision
                ):
                    raise ValueError("Native periodic source identity changed.")
                source_id, source_revision = map_source_id, map_source_revision
            else:
                source_id = self.part.name if source_id is None else source_id
                source_revision = (
                    self.part.part_id if source_revision is None else source_revision
                )
        else:
            if not isinstance(source_map, CellGeometryCoordinateMap):
                raise TypeError(
                    "Result source map does not satisfy its typed identity contract."
                )
            map_source_id, map_source_revision = (
                source_map.source_id,
                source_map.source_revision,
            )
            if (
                source_map.source_map_id != source_map_id
                or (source_id is not None and source_id != map_source_id)
                or (
                    source_revision is not None and source_revision != map_source_revision
                )
            ):
                raise ValueError("Cell geometry source-map identity changed.")
            source_id, source_revision = map_source_id, map_source_revision
        if not source_id or not source_revision:
            raise ValueError("Owner-local geometry requires exact source identities.")
        source_geometry_id = cell_geometry_id(geometry)
        if (
            part_authority.revision_id != self.part.part_id
            or part_authority.structure_id != mesh.topology_id
            or part_authority.semantics_id != source_geometry_id
        ):
            raise ValueError(
                "Accepted part authority does not bind this mesh and exact geometry revision."
            )
        bound_entries = tuple(
            (dependency, accepted_composition.entry(dependency.entry_id))
            for dependency in part_authority.dependencies
        )
        source_mesh_dependencies = tuple(
            (dependency, entry)
            for dependency, entry in bound_entries
            if dependency.facet == "structure"
            and dependency.entry_id != owner_directory_witness.directory_entry_id
        )
        source_mesh_ids = {entry.structure_id for _, entry in source_mesh_dependencies}
        if len(source_mesh_ids) != 1:
            raise ValueError(
                "Accepted part authority must bind exactly one source artifact structure."
            )
        source_mesh_id = next(iter(source_mesh_ids))
        if not any(
            dependency.facet == "revision" and entry.revision_id == source_revision
            for dependency, entry in bound_entries
        ) or (
            source_map_id is not None
            and not any(
                dependency.facet == "semantics" and entry.semantics_id == source_map_id
                for dependency, entry in bound_entries
            )
        ):
            raise ValueError(
                "Accepted part authority lacks source revision/map dependencies."
            )
        from ..discretization._exact_power_geometry import (
            ExactPowerCellGeometrySource,
        )

        if isinstance(source_map, ExactPowerCellGeometrySource):
            context = source_map.periodic_source_context
            source_context_id = None if context is None else context.context_id
        elif isinstance(authority_value, NativePeriodicSource):
            source_context_id = authority_value.binding_id
        else:
            source_context_id = None
        if (
            source_map_id != owner_directory_witness.local_source_map_id
            or source_context_id != owner_directory_witness.local_source_context_id
        ):
            raise ValueError(
                "Owner directory witness does not bind the exact local map/context shard."
            )

        selected = self.local_cell_geometry(rank_, include_halo=include_halo)
        cell_rows = np.asarray(selected["cell_rows"], dtype=np.int64)
        cell_ids = np.asarray(selected["cell_ids"], dtype=np.int64)
        vertex_rows = np.asarray(selected["vertex_rows"], dtype=np.int64)
        vertex_ids = np.asarray(selected["vertex_ids"], dtype=np.int64)
        cell_owner_ranks = np.asarray(self.partition.cell_owner, dtype=np.int32)[
            cell_rows
        ]
        owned = cell_owner_ranks == rank_
        if not np.array_equal(
            cell_ids, np.asarray(owner_directory_witness.local_cell_ids, dtype=np.int64)
        ) or not np.array_equal(
            cell_owner_ranks,
            np.asarray(owner_directory_witness.local_cell_owner_ranks, dtype=np.int32),
        ):
            raise ValueError(
                "Serial cell slice differs from its sharded owner directory."
            )
        source_point_ids = np.asarray(
            owner_directory_witness.local_source_point_ids, dtype=np.int64
        )
        source_owner_ranks = np.asarray(
            owner_directory_witness.local_source_owner_ranks, dtype=np.int32
        )
        source_owned = np.asarray(
            owner_directory_witness.local_source_owned, dtype=np.bool_
        )
        vertex_owner_ranks = owner_directory_witness.owner_ranks_for_ids(
            vertex_ids, directory_kind="mesh-vertices"
        )
        if not np.array_equal(
            vertex_ids,
            np.asarray(owner_directory_witness.local_vertex_ids, dtype=np.int64),
        ) or not np.array_equal(
            vertex_owner_ranks,
            np.asarray(owner_directory_witness.local_vertex_owner_ranks, dtype=np.int32),
        ):
            raise ValueError(
                "Serial vertex slice differs from its sharded owner directory."
            )
        cell_measures_all = np.asarray(
            result.quality.evaluation.measures, dtype=np.float64
        )
        if cell_measures_all.shape != self.cell_global_ids.shape:
            raise ValueError("Certified cell measures do not follow native mesh order.")
        cell_measures = cell_measures_all[cell_rows]
        exact_coordinates = geometry.source_coordinates(selected["coordinate_rows"])
        exact_vertex_coordinates = (
            geometry.source_coordinates(vertex_rows)
            if np.asarray(geometry.coordinates).shape
            == np.asarray(mesh.coordinates).shape
            and np.array_equal(
                np.asarray(geometry.coordinates), np.asarray(mesh.coordinates)
            )
            else None
        )

        topological_dimension = mesh.topological_dimension
        entity_vertices = _entity_vertex_rows(mesh)
        if len(entity_vertices) != topological_dimension + 1:
            raise ValueError("Cell mesh entity vertex rows omit a topology dimension.")
        selected_entity_rows: list[np.ndarray | None] = [
            None for _ in range(topological_dimension + 1)
        ]
        selected_entity_rows[topological_dimension] = cell_rows
        for degree in range(topological_dimension, 0, -1):
            incidence = mesh.topology.incidences[degree - 1].relation
            source_rows = np.asarray(incidence.source_indices, dtype=np.int64)
            target_rows = np.asarray(incidence.target_indices, dtype=np.int64)
            valid_routes = np.asarray(incidence.valid, dtype=np.bool_)
            parent_rows = selected_entity_rows[degree]
            if parent_rows is None:
                raise RuntimeError("Local entity closure lost its parent rows.")
            selected_entity_rows[degree - 1] = np.unique(
                source_rows[
                    valid_routes & np.isin(target_rows, parent_rows, assume_unique=False)
                ]
            )

        entity_set_ids: dict[int, str] = {}
        local_entity_ids: dict[int, np.ndarray] = {}
        entity_vertex_offsets: dict[int, np.ndarray] = {}
        entity_vertex_values: dict[int, np.ndarray] = {}
        for degree, rows_at_degree in enumerate(selected_entity_rows):
            if rows_at_degree is None:
                raise RuntimeError("Local entity closure omitted a topology degree.")
            entity_set = mesh.entity_set(degree)
            global_entity_ids = np.asarray(entity_set.entity_ids, dtype=np.int64)
            active = np.asarray(entity_set.active_mask, dtype=np.bool_)
            if np.any(~active[rows_at_degree]):
                raise ValueError("Local cell closure references an inactive entity.")
            if degree == 0:
                ordered_rows = vertex_rows
                if not np.array_equal(np.sort(rows_at_degree), vertex_rows):
                    raise ValueError(
                        "Local vertex closure differs from compact cell vertices."
                    )
            elif degree == topological_dimension:
                ordered_rows = cell_rows
                if not np.array_equal(global_entity_ids[ordered_rows], cell_ids):
                    raise ValueError(
                        "Local cell entity IDs differ from selected cell IDs."
                    )
            else:
                intermediate_ids = global_entity_ids[rows_at_degree]
                ordered_rows = rows_at_degree[np.argsort(intermediate_ids, kind="stable")]
            local_entity_ids[degree] = global_entity_ids[ordered_rows]
            entity_set_ids[degree] = entity_set.entity_set_id
            offsets = np.zeros((ordered_rows.size + 1,), dtype=np.int64)
            values: list[np.ndarray] = []
            for local_row, entity_row in enumerate(ordered_rows):
                vertex_route = np.asarray(
                    entity_vertices[degree][int(entity_row)], dtype=np.int64
                )
                local_route = np.searchsorted(vertex_rows, vertex_route)
                if np.any(local_route >= vertex_rows.size) or not np.array_equal(
                    vertex_rows[local_route], vertex_route
                ):
                    raise ValueError(
                        "Local entity incidence escapes compact vertex rows."
                    )
                values.append(local_route.astype(np.int32, copy=False))
                offsets[local_row + 1] = offsets[local_row] + local_route.size
            entity_vertex_offsets[degree] = offsets
            entity_vertex_values[degree] = (
                np.concatenate(values).astype(np.int32, copy=False)
                if values
                else np.zeros((0,), dtype=np.int32)
            )

        block_counts = np.asarray(
            [block.cell_count for block in mesh.blocks], dtype=np.int64
        )
        block_offsets = np.concatenate(
            (np.zeros((1,), dtype=np.int64), np.cumsum(block_counts))
        )
        block_index_by_name = {
            block.name: index for index, block in enumerate(mesh.blocks)
        }
        restriction = geometry.restriction_source
        periodic = mesh.periodic_topology
        periodic_cell: PeriodicCell | None = None
        if periodic is not None:
            if not isinstance(periodic.cell, PeriodicCell):
                raise TypeError("Owner-local periodic conversion requires PeriodicCell.")
            periodic_cell = periodic.cell
        periodic_topology_id = None if periodic is None else periodic.periodic_topology_id
        if periodic is None:
            periodic_cell_ids = None
            vertex_representative_ids = None
            vertex_windings = None
            cell_windings = None
            vertex_source_ids = vertex_ids
            vertex_absolute_windings = np.zeros((vertex_ids.size, 0), dtype=np.int64)
            cell_key_offsets = np.arange(cell_ids.size + 1, dtype=np.int64)
            cell_key_source_ids = cell_ids.copy()
            cell_key_relative_windings = np.empty((cell_ids.size, 0), dtype=np.int64)
            cell_key_rows = None
            vertex_shifts = None
            periodic_orders = None
        else:
            degree = topological_dimension
            lifted_start, lifted_stop = map(
                int, np.asarray(periodic.lifted_offsets)[degree : degree + 2]
            )
            quotient_start = int(np.asarray(periodic.quotient_offsets)[degree])
            quotient_cell_rows = (
                np.asarray(periodic.orbit_indices, dtype=np.int64)[
                    lifted_start:lifted_stop
                ]
                - quotient_start
            )
            quotient_ids = np.asarray(
                periodic.quotient.entity_sets[degree].entity_ids, dtype=np.int64
            )
            periodic_cell_ids = quotient_ids[quotient_cell_rows[cell_rows]]
            cell_windings = np.asarray(periodic.orbit_shifts, dtype=np.int64)[
                lifted_start:lifted_stop
            ][cell_rows]
            vertex_representatives = np.asarray(
                periodic.vertex_representatives, dtype=np.int64
            )
            vertex_shifts = np.asarray(periodic.vertex_shifts, dtype=np.int64)
            vertex_representative_ids = np.asarray(mesh.vertex_global_ids)[
                vertex_representatives[vertex_rows]
            ]
            vertex_windings = vertex_shifts[vertex_rows]
            vertex_source_ids = vertex_representative_ids
            vertex_absolute_windings = vertex_windings
            key_offsets = np.asarray(periodic.entity_key_offsets, dtype=np.int64)
            key_values = np.asarray(periodic.entity_key_values, dtype=np.int64)
            periodic_orders = identification_orders(periodic.cell)
            stride = periodic_orders.size + 1
            cell_key_rows = []
            for cell_row in cell_rows:
                key_row = quotient_start + int(quotient_cell_rows[int(cell_row)])
                start, stop = map(int, key_offsets[key_row : key_row + 2])
                key = key_values[start:stop]
                if key.size % stride:
                    raise ValueError(
                        "Periodic quotient cell key has malformed vertex rows."
                    )
                cell_key_rows.append(key.reshape((-1, stride)))
            cell_key_offsets = np.zeros((cell_ids.size + 1,), dtype=np.int64)
            np.cumsum([row.shape[0] for row in cell_key_rows], out=cell_key_offsets[1:])
            flattened_keys = (
                np.concatenate(cell_key_rows, axis=0)
                if cell_key_rows
                else np.zeros((0, stride), dtype=np.int64)
            )
            cell_key_source_ids = flattened_keys[:, 0]
            cell_key_relative_windings = flattened_keys[:, 1:]

        elements, _, _ = geometry.resolve(mesh)
        local_blocks: list[dict[str, Any]] = []
        for block_record in selected["blocks"]:
            block_index = block_index_by_name[str(block_record["name"])]
            block = mesh.blocks[block_index]
            local_rows = np.asarray(block_record["cell_rows"], dtype=np.int64)
            block_ids = np.asarray(block_record["cell_ids"], dtype=np.int64)
            global_rows = int(block_offsets[block_index]) + local_rows
            element = elements[block_index]
            chart_arguments = None
            if isinstance(element, PerCellRestrictedCellGeometryElement):
                source_cell_ids = np.asarray(element.source_cell_ids, dtype=np.int64)[
                    local_rows
                ]
                chart_arguments = tuple(
                    element.chart_arguments(int(row)) for row in local_rows
                )
                element = PerCellRestrictedCellGeometryElement(
                    element.source_element,
                    element.cell_kind,
                    block_ids,
                    source_cell_ids,
                    chart_arguments,
                )
                element_rows = np.arange(block_ids.size, dtype=np.int32)
            elif restriction is not None:
                source_index = restriction.block_names.index(block.name)
                source_cell_ids = np.asarray(
                    restriction.parent_cell_ids[source_index], dtype=np.int64
                )[local_rows]
                element_rows = local_rows.astype(np.int32, copy=False)
            else:
                source_cell_ids = np.array(block_ids, dtype=np.int64, copy=True)
                element_rows = local_rows.astype(np.int32, copy=False)
            local_block = {
                **block_record,
                "element": element,
                "element_rows": element_rows,
                "source_cell_ids": source_cell_ids,
                "cell_measures": cell_measures_all[global_rows],
                "chart_arguments": chart_arguments,
                "cell_vertex_shifts": None,
            }
            if periodic is not None:
                if vertex_windings is None or vertex_shifts is None:
                    raise ValueError("Periodic local cell shifts are missing.")
                local_vertices = np.asarray(block.vertices, dtype=np.int64)[local_rows]
                local_valid = np.asarray(block.vertex_valid, dtype=np.bool_)[local_rows]
                shifts = np.zeros(
                    (local_rows.size, block.arity, vertex_windings.shape[1]),
                    dtype=np.int64,
                )
                for local_row in range(local_rows.size):
                    active_vertices = local_vertices[local_row, local_valid[local_row]]
                    absolute = vertex_shifts[active_vertices]
                    shifts[local_row, local_valid[local_row]] = absolute
                local_block["cell_vertex_shifts"] = shifts
            if block.cell_kind == "polyhedron":
                connectivity = mesh.connectivity
                if not isinstance(connectivity, PolyhedralConnectivity):
                    raise TypeError("Polyhedron blocks require packed face incidence.")
                connectivity_cell_ids = np.asarray(
                    connectivity.cell_global_ids, dtype=np.int64
                )
                connectivity_row_by_id = {
                    int(identifier): row
                    for row, identifier in enumerate(connectivity_cell_ids)
                }
                selected_connectivity_rows = np.asarray(
                    [connectivity_row_by_id[int(identifier)] for identifier in block_ids],
                    dtype=np.int64,
                )
                cell_face_offsets_source = np.asarray(
                    connectivity.cell_face_offsets, dtype=np.int64
                )
                cell_face_values_source = np.asarray(
                    connectivity.cell_face_values, dtype=np.int64
                )
                cell_face_signs_source = np.asarray(
                    connectivity.cell_face_sign_values, dtype=np.int8
                )
                selected_faces = []
                selected_signs = []
                for connectivity_row in selected_connectivity_rows:
                    start, stop = map(
                        int,
                        cell_face_offsets_source[connectivity_row : connectivity_row + 2],
                    )
                    selected_faces.append(cell_face_values_source[start:stop])
                    selected_signs.append(cell_face_signs_source[start:stop])
                face_rows = (
                    np.unique(np.concatenate(selected_faces))
                    if selected_faces
                    else np.zeros((0,), dtype=np.int64)
                )
                face_ids_source = np.asarray(connectivity.face_global_ids, dtype=np.int64)
                face_rows = face_rows[
                    np.argsort(face_ids_source[face_rows], kind="stable")
                ]
                face_row_to_local = {
                    int(face_row): local_face
                    for local_face, face_row in enumerate(face_rows)
                }
                local_cell_face_offsets = np.zeros(
                    (len(selected_faces) + 1,), dtype=np.int64
                )
                local_cell_face_values: list[int] = []
                local_cell_face_signs: list[int] = []
                for cell, (faces, signs) in enumerate(
                    zip(selected_faces, selected_signs, strict=True)
                ):
                    local_cell_face_values.extend(
                        face_row_to_local[int(face)] for face in faces
                    )
                    local_cell_face_signs.extend(int(value) for value in signs)
                    local_cell_face_offsets[cell + 1] = len(local_cell_face_values)
                face_vertex_offsets_source = np.asarray(
                    connectivity.face_vertex_offsets, dtype=np.int64
                )
                face_vertex_values_source = np.asarray(
                    connectivity.face_vertex_values, dtype=np.int64
                )
                local_face_vertex_offsets = np.zeros(
                    (face_rows.size + 1,), dtype=np.int64
                )
                local_face_vertex_values: list[int] = []
                for local_face, face_row in enumerate(face_rows):
                    start, stop = map(
                        int, face_vertex_offsets_source[face_row : face_row + 2]
                    )
                    global_vertices = face_vertex_values_source[start:stop]
                    local_vertices = np.searchsorted(vertex_rows, global_vertices)
                    if np.any(local_vertices >= vertex_rows.size) or not np.array_equal(
                        vertex_rows[local_vertices], global_vertices
                    ):
                        raise ValueError("Polyhedron face escapes local vertex rows.")
                    local_face_vertex_values.extend(
                        int(value) for value in local_vertices
                    )
                    local_face_vertex_offsets[local_face + 1] = len(
                        local_face_vertex_values
                    )
                local_block.update(
                    {
                        "face_ids": face_ids_source[face_rows],
                        "cell_face_offsets": local_cell_face_offsets,
                        "cell_face_values": np.asarray(
                            local_cell_face_values, dtype=np.int64
                        ),
                        "cell_face_signs": np.asarray(
                            local_cell_face_signs, dtype=np.int8
                        ),
                        "face_vertex_offsets": local_face_vertex_offsets,
                        "face_vertex_values": np.asarray(
                            local_face_vertex_values, dtype=np.int32
                        ),
                    }
                )
            local_blocks.append(local_block)

        return {
            "part_name": self.part.name,
            "part_id": self.part.part_id,
            "distribution_id": owner_directory_witness.directory_id,
            "source_id": source_id,
            "source_revision": source_revision,
            "source_mesh_id": source_mesh_id,
            "source_topology_id": mesh.topology_id,
            "source_geometry_id": source_geometry_id,
            "source_map_id": source_map_id,
            "source_context_id": source_context_id,
            "geometry_layout_id": geometry.geometry_layout_id,
            "coordinate_contract_id": result.coordinate_contract.spatial_id,
            "coordinate_contract": result.coordinate_contract,
            "topological_dimension": topological_dimension,
            "part_authority": part_authority,
            "accepted_composition": accepted_composition,
            "acceptance_receipt": acceptance_receipt,
            "owner_directory_witness": owner_directory_witness,
            "rank": rank_,
            "cell_rows": cell_rows,
            "cell_ids": cell_ids,
            "owned": owned,
            "cell_measures": cell_measures,
            "vertex_rows": vertex_rows,
            "vertex_ids": vertex_ids,
            "vertex_coordinates": np.asarray(selected["vertex_coordinates"]),
            "exact_vertex_coordinates": exact_vertex_coordinates,
            "coordinate_rows": np.asarray(selected["coordinate_rows"], dtype=np.int64),
            "source_point_ids": source_point_ids,
            "source_owner_ranks": source_owner_ranks,
            "source_owned": source_owned,
            "vertex_owner_ranks": vertex_owner_ranks,
            "coordinates": np.asarray(selected["coordinates"], dtype=np.float64),
            "exact_coordinates": exact_coordinates,
            "entity_set_ids": entity_set_ids,
            "local_entity_ids": local_entity_ids,
            "entity_vertex_offsets": entity_vertex_offsets,
            "entity_vertex_values": entity_vertex_values,
            "cell_key_offsets": cell_key_offsets,
            "cell_key_source_ids": cell_key_source_ids,
            "cell_key_relative_windings": cell_key_relative_windings,
            "cell_owner_ranks": cell_owner_ranks,
            "support_cell_mask": np.zeros((cell_ids.size,), dtype=np.bool_),
            "vertex_source_ids": vertex_source_ids,
            "vertex_absolute_windings": vertex_absolute_windings,
            "candidate_revision_id": None,
            "certified_periodic_margin": None,
            "periodic_shell_certificate": None,
            "periodic_lift_matrix": (
                None if periodic_cell is None else np.asarray(periodic_cell.vectors)
            ),
            "periodic_lift_id": (
                None if periodic_cell is None else periodic_cell.cell_id
            ),
            "coordinates_are_lifted": True,
            "serial_source_view": True,
            "periodic_topology_id": periodic_topology_id,
            "periodic_cell_ids": periodic_cell_ids,
            "vertex_representative_ids": vertex_representative_ids,
            "vertex_windings": vertex_windings,
            "cell_windings": cell_windings,
            "blocks": tuple(local_blocks),
        }

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


def _balance_feasible(
    owner: np.ndarray, weights: np.ndarray, part_count: int, target: float, /
) -> bool:
    part_weights = np.bincount(owner, weights=weights, minlength=part_count)
    if part_weights.size != part_count or np.any(part_weights == 0):
        return False
    mean = float(np.sum(weights, dtype=np.float64)) / part_count
    return float(np.max(part_weights)) <= target * mean * (
        1.0 + 8.0 * np.finfo(np.float64).eps
    )


def _partition_owners(
    part: MeshPart,
    weights: np.ndarray,
    policy: MeshPartitionPolicy,
    ownership: ArrayLike | None,
    /,
) -> tuple[np.ndarray, str, bool]:
    if ownership is not None and policy.kind is not MeshPartitionKind.PROVIDER:
        raise ValueError("Supplied ownership requires the PROVIDER partition route.")
    periodic_mesh, periodic_layout = _periodic_layout(part)
    partition_weights = weights
    cell_ids = _native_cell_ids(part)
    if periodic_layout is not None:
        partition_weights, _ = _periodic_quotient_weights(weights, periodic_layout)
        cell_ids = periodic_layout.entity_ids
    match policy.kind:
        case MeshPartitionKind.MORTON | MeshPartitionKind.HILBERT:
            if periodic_layout is None:
                centroids = _cell_centroids(part)
            else:
                periodic_mesh = _require_periodic_mesh(periodic_mesh)
                centroids = _periodic_quotient_cell_centroids(
                    periodic_mesh, periodic_layout
                )
            codes = _curve_codes(centroids, policy.kind, policy.curve_depth)
            order = np.lexsort((cell_ids, codes))
            quotient_owner = _contiguous_ranges(
                order, partition_weights, policy.part_count
            )
            provenance = f"{policy.kind.value}-curve-weighted-ranges"
            owner = (
                quotient_owner
                if periodic_layout is None
                else _periodic_expand_owners(quotient_owner, periodic_layout)
            )
            return (
                owner,
                provenance,
                _balance_feasible(
                    quotient_owner,
                    partition_weights,
                    policy.part_count,
                    policy.maximum_imbalance,
                ),
            )
        case MeshPartitionKind.GRAPH:
            if policy.part_count > partition_weights.size:
                raise ValueError("part_count exceeds the number of cells.")
            if periodic_layout is None:
                adjacency = _carrier_adjacency(part)
            else:
                periodic_mesh = _require_periodic_mesh(periodic_mesh)
                adjacency = _periodic_quotient_adjacency(periodic_mesh)
            graph = WeightedCSRGraph(
                adjacency.offsets,
                adjacency.neighbors,
                vertex_weights=partition_weights,
                entity_ids=cell_ids,
            )
            result = partition_graph(
                graph,
                GraphPartitionPlan(
                    policy.part_count,
                    maximum_imbalance=policy.maximum_imbalance,
                    require_nonempty=True,
                ),
            )
            quotient_owner = np.asarray(result.owners, dtype=np.int32)
            owner = (
                quotient_owner
                if periodic_layout is None
                else _periodic_expand_owners(quotient_owner, periodic_layout)
            )
            return owner, result.provenance, result.feasible
        case MeshPartitionKind.METIS:
            if policy.part_count > partition_weights.size:
                raise ValueError("part_count exceeds the number of cells.")
            if periodic_layout is None:
                adjacency = _carrier_adjacency(part)
            else:
                periodic_mesh = _require_periodic_mesh(periodic_mesh)
                adjacency = _periodic_quotient_adjacency(periodic_mesh)
            quotient_owner = metis_partition(
                np.asarray(adjacency.offsets),
                np.asarray(adjacency.neighbors),
                partition_weights,
                policy.part_count,
                imbalance=policy.maximum_imbalance,
                seed=policy.graph_seed,
            )
            if np.unique(quotient_owner).size != policy.part_count:
                raise MetisPartitionError("METIS left at least one part empty.")
            owner = (
                quotient_owner
                if periodic_layout is None
                else _periodic_expand_owners(quotient_owner, periodic_layout)
            )
            return (
                owner,
                metis_identity(),
                _balance_feasible(
                    quotient_owner,
                    partition_weights,
                    policy.part_count,
                    policy.maximum_imbalance,
                ),
            )
        case MeshPartitionKind.PROVIDER:
            if ownership is None:
                raise ValueError("The PROVIDER partition route requires ownership.")
            owner = np.asarray(ownership)
            if owner.shape != weights.shape or owner.dtype.kind not in "iu":
                raise ValueError("Provider ownership must be an integer per native cell.")
            quotient_owner = (
                owner
                if periodic_layout is None
                else _periodic_collapse_owners(owner, periodic_layout)
            )
            if periodic_layout is not None:
                owner = _periodic_expand_owners(quotient_owner, periodic_layout)
            return (
                owner,
                "provider-ownership",
                _balance_feasible(
                    quotient_owner,
                    partition_weights,
                    policy.part_count,
                    policy.maximum_imbalance,
                ),
            )
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
    GRAPH is always the native meshcore partition; METIS is an explicit
    comparison route and is never selected as a fallback.
    """
    if not isinstance(part, MeshPart):
        raise TypeError("part must be MeshPart.")
    if not isinstance(policy, MeshPartitionPolicy):
        raise TypeError("policy must be MeshPartitionPolicy.")
    weights = _cell_weights(cell_weights, _native_cell_ids(part).size)
    owner, provenance, feasible = _partition_owners(part, weights, policy, ownership)
    return MeshDistribution(
        part,
        CellPartition(owner, policy.part_count),
        halo_width=policy.halo_width,
        cell_weights=weights,
        partition_kind=policy.kind,
        partition_provenance=provenance,
        partition_feasible=feasible,
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
    """Carry inherited ownership on the lifted or periodic quotient complex."""
    parts = policy.part_count
    source_mesh = _periodic_mesh(source.part)
    target_mesh = _periodic_mesh(target_part)
    if source_mesh is None or target_mesh is None:
        raise TypeError(
            "Distribution ownership transitions require certified cell parts."
        )
    source_layout = _periodic_cell_layout(source_mesh)
    target_layout = _periodic_cell_layout(target_mesh)
    if (source_layout is None) != (target_layout is None):
        raise ValueError(
            "Distribution transitions cannot change periodic quotient topology."
        )
    if policy.kind is MeshPartitionKind.PROVIDER:
        owner, provenance, _ = _partition_owners(target_part, weights, policy, ownership)
        return owner, True, f"transition-{provenance}"
    if source_layout is None or target_layout is None:
        inherited, _ = inherit_cell_owners(
            np.asarray(source.partition.cell_owner), routes[0], routes[1], weights.size
        )
        grown = _grow_ownership(inherited, _carrier_adjacency(target_part))
        if np.all(grown >= 0):
            part_weights = np.bincount(grown, weights=weights, minlength=parts)
            if (
                np.all(part_weights > 0.0)
                and np.max(part_weights) / np.mean(part_weights)
                <= policy.maximum_imbalance
            ):
                return grown, False, "transition-inherited"
        fresh, provenance, _ = _partition_owners(target_part, weights, policy, ownership)
        relabeled = _relabel_to_inherited(fresh, inherited, weights, parts)
        return relabeled, True, f"transition-rebalanced:{provenance}"
    source_owner = _periodic_collapse_owners(
        np.asarray(source.partition.cell_owner), source_layout
    )
    quotient_weights, _ = _periodic_quotient_weights(weights, target_layout)
    source_orbits = source_layout.orbit_rows[routes[0]]
    target_orbits = target_layout.orbit_rows[routes[1]]
    orbit_routes = np.unique(np.stack((source_orbits, target_orbits), axis=1), axis=0)
    inherited, _ = inherit_cell_owners(
        source_owner,
        orbit_routes[:, 0],
        orbit_routes[:, 1],
        target_layout.entity_ids.size,
    )
    grown = _grow_ownership(inherited, _periodic_quotient_adjacency(target_mesh))
    if np.all(grown >= 0):
        part_weights = np.bincount(grown, weights=quotient_weights, minlength=parts)
        if (
            np.all(part_weights > 0.0)
            and np.max(part_weights) / np.mean(part_weights) <= policy.maximum_imbalance
        ):
            return (
                _periodic_expand_owners(grown, target_layout),
                False,
                "transition-inherited",
            )
    fresh, provenance, _ = _partition_owners(target_part, weights, policy, ownership)
    fresh_quotient = _periodic_collapse_owners(fresh, target_layout)
    relabeled = _relabel_to_inherited(fresh_quotient, inherited, quotient_weights, parts)
    return (
        _periodic_expand_owners(relabeled, target_layout),
        True,
        f"transition-rebalanced:{provenance}",
    )


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
    ) -> None:
        if not isinstance(source, MeshDistribution) or not isinstance(
            target, MeshDistribution
        ):
            raise TypeError("Transition endpoints must be MeshDistribution values.")
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


@final
class OwnerLocalGraphPartitionEvidence(StrictModule, NonTrainableState):
    """Graph labels bound to one accepted owner-local periodic view."""

    result: DistributedGraphPartitionResult
    source_view_id: str = eqx.field(static=True)
    periodic_face_coverage_id: str = eqx.field(static=True)
    owner_directory_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        result: DistributedGraphPartitionResult,
        source_view_id: str,
        periodic_face_coverage_id: str,
        owner_directory_id: str,
        /,
    ) -> None:
        from ..graph._distributed_partition import DistributedGraphPartitionResult

        identities = (
            source_view_id,
            periodic_face_coverage_id,
            owner_directory_id,
        )
        if not isinstance(result, DistributedGraphPartitionResult) or any(
            not isinstance(identity, str) or not identity for identity in identities
        ):
            raise TypeError(
                "Owner-local graph evidence requires typed result and view identities."
            )
        self.result = result
        self.source_view_id = source_view_id
        self.periodic_face_coverage_id = periodic_face_coverage_id
        self.owner_directory_id = owner_directory_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "owner-local-periodic-graph-partition",
                "source_view": source_view_id,
                "periodic_face_coverage": periodic_face_coverage_id,
                "owner_directory": owner_directory_id,
                "graph": result.graph_id,
                "plan": result.plan_id,
                "migration": result.migration_id,
                "result": result.result_id,
            }
        )


def _partition_owner_local_periodic_candidate(
    candidate: OwnerLocalPeriodicCandidate,
    plan: GraphPartitionPlan,
    execution_group: ExecutionGroup,
    /,
    *,
    axis_name: str,
) -> DistributedGraphPartitionResult:
    """Partition the exact owned CSR rows of a periodic Delaunay candidate."""
    import jax

    from .._execution_runtime import ExecutionGroup
    from ..graph._distributed_partition import (
        DistributedGraphPartitionResult,
        DistributedWeightedCSRGraph,
        partition_graph_distributed,
    )
    from ..graph._partition import GraphPartitionPlan
    from ..meshing._periodic import OwnerLocalPeriodicCandidate

    if (
        not isinstance(candidate, OwnerLocalPeriodicCandidate)
        or not isinstance(plan, GraphPartitionPlan)
        or not isinstance(execution_group, ExecutionGroup)
    ):
        raise TypeError(
            "Periodic graph partition requires a typed candidate, plan, and group."
        )
    members = execution_group.spec.process_indices
    try:
        rank = members.index(jax.process_index())
    except ValueError as error:
        raise ValueError(
            "The current process is outside the periodic graph group."
        ) from error
    if (
        not execution_group.is_member
        or tuple(device.process_index for device in execution_group.devices) != members
        or rank != candidate.rank
        or len(members) != candidate.part_count
        or plan.part_count != candidate.part_count
        or not isinstance(axis_name, str)
        or not axis_name.strip()
    ):
        raise ValueError("Periodic candidate, graph plan, and execution group differ.")
    # No heterogeneous cell-work model is declared; balance one integer unit per owned cell.
    owned_cell_work_weights = np.ones(
        np.asarray(candidate.owned_cell_ids).shape, dtype=np.int64
    )
    graph = DistributedWeightedCSRGraph(
        candidate.cell_neighbor_offsets,
        candidate.cell_neighbor_ids,
        entity_ids=candidate.owned_cell_ids,
        owner_ranks=candidate.owned_cell_owner_ranks,
        neighbor_owner_ranks=candidate.cell_neighbor_owner_ranks,
        edge_weights=candidate.cell_neighbor_weights,
        vertex_weights=owned_cell_work_weights,
    )
    result = partition_graph_distributed(
        graph, plan, execution_group, axis_name=axis_name
    )
    if (
        not isinstance(result, DistributedGraphPartitionResult)
        or not np.array_equal(
            np.asarray(result.entity_ids, dtype=np.int64),
            np.asarray(candidate.owned_cell_ids, dtype=np.int64),
        )
        or np.asarray(result.owners).shape != np.asarray(candidate.owned_cell_ids).shape
        or result.plan_id != plan.plan_id
        or not result.feasible
    ):
        raise ValueError(
            "Periodic graph partition did not produce feasible owner labels."
        )
    return result


def _owner_local_keyed_target_owners(
    source_keys: np.ndarray,
    target_keys: np.ndarray,
    target_owner_ranks: np.ndarray,
    rank: int,
    execution_group: ExecutionGroup,
    key_identity_id: str,
    axis_name: str,
    /,
    *,
    require_exact_coverage: bool = True,
    allow_duplicate_queries: bool = False,
) -> np.ndarray:
    """Resolve target owners for local source keys through sharded key buckets."""
    parts = len(execution_group.spec.process_indices)
    key_width = source_keys.shape[1]
    message_width = key_width + 1
    target_outgoing_rows: list[list[np.ndarray]] = [[] for _ in range(parts)]
    local_target_rows: list[np.ndarray] = []
    local_error = None
    try:
        for key_row, owner_value in zip(target_keys, target_owner_ranks, strict=True):
            key = tuple(int(value) for value in key_row)
            bucket = int(
                canonical_fingerprint({"key_identity": key_identity_id, "key": key})[:2],
                16,
            )
            message = np.empty((message_width,), dtype=np.int64)
            message[:-1] = key_row
            message[-1] = int(owner_value)
            destination = bucket % parts
            if destination == rank:
                local_target_rows.append(message)
            else:
                target_outgoing_rows[destination].append(message)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_target_encode",
        "keyed target owner directory",
    )
    target_outgoing = tuple(
        np.stack(rows) if rows else np.zeros((0, message_width), dtype=np.int64)
        for rows in target_outgoing_rows
    )
    target_incoming = _owner_exchange_message_banks(
        target_outgoing, rank, execution_group, f"{axis_name}_target_directory"
    )

    source_outgoing_rows: list[list[np.ndarray]] = [[] for _ in range(parts)]
    local_source_rows: list[np.ndarray] = []
    local_error = None
    try:
        for row, key_row in enumerate(source_keys):
            key = tuple(int(value) for value in key_row)
            bucket = int(
                canonical_fingerprint({"key_identity": key_identity_id, "key": key})[:2],
                16,
            )
            message = np.empty((message_width,), dtype=np.int64)
            message[:-1] = key_row
            message[-1] = row
            destination = bucket % parts
            if destination == rank:
                local_source_rows.append(message)
            else:
                source_outgoing_rows[destination].append(message)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_source_encode",
        "keyed source owner lookup",
    )
    source_outgoing = tuple(
        np.stack(rows) if rows else np.zeros((0, message_width), dtype=np.int64)
        for rows in source_outgoing_rows
    )
    source_incoming = _owner_exchange_message_banks(
        source_outgoing, rank, execution_group, f"{axis_name}_source_queries"
    )

    target_owners: dict[tuple[int, ...], int] = {}
    source_queries: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    local_error = None
    try:
        for row in local_target_rows:
            key = tuple(int(value) for value in row[:-1])
            owner = int(row[-1])
            if not 0 <= owner < parts or key in target_owners:
                raise ValueError(
                    "Keyed target owner directory has duplicate or invalid rows."
                )
            target_owners[key] = owner
        for rows in target_incoming.values():
            bank = np.asarray(rows, dtype=np.int64)
            if bank.ndim != 2 or bank.shape[1] != message_width:
                raise ValueError("Keyed target owner messages have the wrong layout.")
            for row in bank:
                key = tuple(int(value) for value in row[:-1])
                owner = int(row[-1])
                if not 0 <= owner < parts or key in target_owners:
                    raise ValueError(
                        "Keyed target owner directory has duplicate or invalid rows."
                    )
                target_owners[key] = owner
        for row in local_source_rows:
            key = tuple(int(value) for value in row[:-1])
            source_row = int(row[-1])
            if not 0 <= source_row < source_keys.shape[0]:
                raise ValueError("Keyed source lookup contains an invalid local row.")
            requests = source_queries.setdefault(key, [])
            if requests and not allow_duplicate_queries:
                raise ValueError("Keyed source lookup contains duplicate entity keys.")
            requests.append((rank, source_row))
        for sender, rows in source_incoming.items():
            bank = np.asarray(rows, dtype=np.int64)
            if bank.ndim != 2 or bank.shape[1] != message_width:
                raise ValueError("Keyed source lookup messages have the wrong layout.")
            for row in bank:
                key = tuple(int(value) for value in row[:-1])
                source_row = int(row[-1])
                if source_row < 0:
                    raise ValueError("Keyed source lookup contains an invalid local row.")
                requests = source_queries.setdefault(key, [])
                if requests and not allow_duplicate_queries:
                    raise ValueError(
                        "Keyed source lookup contains duplicate entity keys."
                    )
                requests.append((int(sender), source_row))
        if require_exact_coverage:
            if target_owners.keys() != source_queries.keys():
                raise ValueError(
                    "Keyed source and target owner maps do not cover the same entity IDs."
                )
        elif not source_queries.keys() <= target_owners.keys():
            raise ValueError(
                "A requested keyed owner is absent from the target directory."
            )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_lookup",
        "keyed source/target owner bijection",
    )

    reply_rows: list[list[np.ndarray]] = [[] for _ in range(parts)]
    owners = np.full((source_keys.shape[0],), -1, dtype=np.int32)
    local_error = None
    try:
        for key, requests in source_queries.items():
            owner = target_owners[key]
            for source_rank, source_row in requests:
                if not 0 <= source_row < owners.size or owners[source_row] >= 0:
                    raise ValueError(
                        "Keyed target owner query rows are duplicated or invalid."
                    )
                if source_rank == rank:
                    owners[source_row] = owner
                else:
                    reply_rows[source_rank].append(
                        np.asarray((source_row, owner), dtype=np.int64)
                    )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
        reply_rows = [[] for _ in range(parts)]
        owners.fill(-1)
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_reply_encode",
        "keyed target owner replies",
    )
    replies_outgoing = tuple(
        np.stack(rows) if rows else np.zeros((0, 2), dtype=np.int64)
        for rows in reply_rows
    )
    replies_incoming = _owner_exchange_message_banks(
        replies_outgoing, rank, execution_group, f"{axis_name}_owner_replies"
    )
    local_error = None
    try:
        for rows in replies_incoming.values():
            bank = np.asarray(rows, dtype=np.int64)
            if bank.ndim != 2 or bank.shape[1] != 2:
                raise ValueError("Keyed target owner replies have the wrong layout.")
            for source_row_value, target_owner_value in bank:
                source_row = int(source_row_value)
                target_owner = int(target_owner_value)
                if (
                    not 0 <= source_row < owners.size
                    or owners[source_row] >= 0
                    or not 0 <= target_owner < parts
                ):
                    raise ValueError(
                        "Keyed target owner replies are duplicated or invalid."
                    )
                owners[source_row] = target_owner
        if np.any(owners < 0):
            raise ValueError("Keyed target owner lookup omitted source rows.")
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_owner_validation",
        "keyed target owner lookup",
    )
    return owners


def _owner_local_state_payload(
    value: ArrayLike, /
) -> tuple[Array | np.ndarray, np.ndarray, str | None, object | None]:
    """Return exact storage rows and preserve typed PRNG implementation metadata."""
    import jax

    if isinstance(value, jax.Array) and jax.dtypes.issubdtype(
        value.dtype, jax.dtypes.prng_key
    ):
        implementation = jax.random.key_impl(value)
        implementation_id = str(implementation)
        data = np.asarray(jax.random.key_data(value))
        if data.dtype.kind not in "iu" or not data.dtype.isnative:
            raise TypeError("Typed PRNG key data must be native integer words.")
        return value, data, implementation_id, implementation
    data = np.asarray(value)
    return data, data, None, None


def _owner_local_state_value_identity(value: ArrayLike, /) -> str:
    array, payload, implementation_id, _ = _owner_local_state_payload(value)
    if implementation_id is not None:
        return canonical_fingerprint(
            {
                "kind": "typed-prng-key-array",
                "shape": tuple(int(size) for size in array.shape),
                "implementation": implementation_id,
                "key_data": array_tree_fingerprint(payload),
            }
        )
    return canonical_fingerprint(array_tree_fingerprint(payload))


@final
class OwnerLocalGraphMigrationEvidence(StrictModule, NonTrainableState):
    """Bind an applied graph partition to its source and migrated local cells."""

    source_candidate_id: str = eqx.field(static=True)
    source_part_id: str = eqx.field(static=True)
    periodic_face_coverage_id: str = eqx.field(static=True)
    source_directory_id: str = eqx.field(static=True)
    target_part_id: str = eqx.field(static=True)
    target_directory_id: str = eqx.field(static=True)
    group_id: str = eqx.field(static=True)
    local_rank: int = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    result: DistributedGraphPartitionResult
    target_cell_ids: Array
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        result: DistributedGraphPartitionResult,
        source_candidate_id: str,
        source_part_id: str,
        periodic_face_coverage_id: str,
        source_directory_id: str,
        target_part_id: str,
        target_directory_id: str,
        group_id: str,
        local_rank: int,
        part_count: int,
        target_cell_ids: ArrayLike,
        /,
    ) -> None:
        from ..graph._distributed_partition import DistributedGraphPartitionResult

        if not isinstance(result, DistributedGraphPartitionResult):
            raise TypeError("result must be DistributedGraphPartitionResult.")
        identities = (
            source_candidate_id,
            source_part_id,
            periodic_face_coverage_id,
            source_directory_id,
            target_part_id,
            target_directory_id,
            group_id,
        )
        raw_ids = np.asarray(target_cell_ids)
        result_ids = np.asarray(result.entity_ids, dtype=np.int64)
        result_owners = np.asarray(result.owners, dtype=np.int64)
        if (
            any(not isinstance(value, str) or not value for value in identities)
            or isinstance(local_rank, bool)
            or not isinstance(local_rank, (int, np.integer))
            or isinstance(part_count, bool)
            or not isinstance(part_count, (int, np.integer))
            or part_count < 1
            or not 0 <= local_rank < part_count
            or not result.feasible
            or result_ids.ndim != 1
            or result_owners.shape != result_ids.shape
            or np.unique(result_ids).size != result_ids.size
            or np.any(result_owners < 0)
            or np.any(result_owners >= part_count)
            or raw_ids.dtype.kind not in "iu"
            or raw_ids.ndim != 1
            or (
                raw_ids.dtype.kind == "u"
                and raw_ids.size
                and int(np.max(raw_ids)) > np.iinfo(np.int64).max
            )
            or (
                raw_ids.size
                and (np.any(raw_ids < 0) or np.unique(raw_ids).size != raw_ids.size)
            )
        ):
            raise ValueError("Owner-local graph migration evidence is malformed.")
        canonical_ids = tuple(int(value) for value in np.sort(raw_ids))
        self.source_candidate_id = source_candidate_id
        self.source_part_id = source_part_id
        self.periodic_face_coverage_id = periodic_face_coverage_id
        self.source_directory_id = source_directory_id
        self.target_part_id = target_part_id
        self.target_directory_id = target_directory_id
        self.group_id = group_id
        self.local_rank = int(local_rank)
        self.part_count = int(part_count)
        self.result = result
        self.target_cell_ids = jnp.asarray(canonical_ids, dtype=jnp.int64)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "owner-local-graph-migration",
                "source_candidate": source_candidate_id,
                "source_part": source_part_id,
                "source_face_coverage": periodic_face_coverage_id,
                "source_directory": source_directory_id,
                "target_part": target_part_id,
                "target_directory": target_directory_id,
                "group": group_id,
                "local_rank": int(local_rank),
                "part_count": int(part_count),
                "graph_result": result.result_id,
                "target_cell_ids": canonical_ids,
            }
        )

    def collective_status(
        self,
        local_success: ArrayLike,
        execution_group: ExecutionGroup,
        /,
        *,
        axis_name: str,
        phase: str,
        failure_reason: str = "Owner-local graph state preparation failed.",
    ) -> bool:
        import jax

        if not isinstance(execution_group, ExecutionGroup):
            raise TypeError("execution_group must be ExecutionGroup.")
        members = execution_group.spec.process_indices
        try:
            rank = members.index(jax.process_index())
        except ValueError as error:
            raise ValueError(
                "The current process is outside the graph migration group."
            ) from error
        local_error = None
        try:
            success = np.asarray(local_success)
            if (
                execution_group.spec.group_id != self.group_id
                or len(members) != self.part_count
                or rank != self.local_rank
                or success.shape != ()
                or success.dtype != np.dtype(np.bool_)
                or not isinstance(axis_name, str)
                or not axis_name.strip()
                or not isinstance(phase, str)
                or not phase.strip()
                or not isinstance(failure_reason, str)
                or not failure_reason.strip()
            ):
                raise ValueError("Owner-local graph collective status is malformed.")
            if not bool(success):
                local_error = failure_reason
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            axis_name,
            phase,
        )
        return True

    def transfer_keyed_values(
        self,
        source_entry: CompositionEntry,
        source_entity_keys: ArrayLike,
        source_owner_ranks: ArrayLike,
        source_values: ArrayLike,
        target_entity_keys: ArrayLike,
        target_owner_ranks: ArrayLike,
        execution_group: ExecutionGroup,
        /,
        *,
        target_structure_id: str,
        key_identity_id: str,
        axis_name: str = "owner_local_graph_keyed_state",
        value_fiber_shape: tuple[int, ...] | None = None,
    ) -> OwnerLocalGraphKeyedStateTransfer:
        """Migrate exact owner rows over the keyed graph directory.

        ``value_fiber_shape`` declares the logical trailing shape of each row;
        use ``()`` for a scalar row, whose wire fiber has width one. If omitted,
        the source payload's declared array tail is preserved.
        """
        return _transfer_owner_local_graph_keyed_values(
            self,
            source_entry,
            source_entity_keys,
            source_owner_ranks,
            source_values,
            target_entity_keys,
            target_owner_ranks,
            execution_group,
            target_structure_id=target_structure_id,
            key_identity_id=key_identity_id,
            axis_name=axis_name,
            value_fiber_shape=value_fiber_shape,
        )


@final
class OwnerLocalGraphKeyedStateTransfer(StrictModule, NonTrainableState):
    """Exact owner-local keyed-row migration and lifecycle ownership evidence."""

    migration_evidence_id: str = eqx.field(static=True)
    group_id: str = eqx.field(static=True)
    local_rank: int = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    target_directory_id: str = eqx.field(static=True)
    source_entry_id: str = eqx.field(static=True)
    source_entry_record_id: str = eqx.field(static=True)
    source_structure_id: str = eqx.field(static=True)
    source_owner_id: str = eqx.field(static=True)
    source_role: CompositionRole = eqx.field(static=True)
    source_revision_id: str = eqx.field(static=True)
    source_semantics_id: str = eqx.field(static=True)
    target_structure_id: str = eqx.field(static=True)
    key_identity_id: str = eqx.field(static=True)
    route_identity_id: str = eqx.field(static=True)
    source_key_set_id: str = eqx.field(static=True)
    target_key_set_id: str = eqx.field(static=True)
    source_owner_map_id: str = eqx.field(static=True)
    target_owner_map_id: str = eqx.field(static=True)
    source_value_id: str = eqx.field(static=True)
    target_value_id: str = eqx.field(static=True)
    source_content_root: str = eqx.field(static=True)
    target_content_root: str = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)
    target_entity_keys: Array
    target_values: Array
    successful: Array

    def __init__(
        self,
        migration_evidence_id: str,
        group_id: str,
        local_rank: int,
        part_count: int,
        target_directory_id: str,
        source_entry_id: str,
        source_entry_record_id: str,
        source_structure_id: str,
        source_owner_id: str,
        source_role: CompositionRole,
        source_revision_id: str,
        source_semantics_id: str,
        target_structure_id: str,
        key_identity_id: str,
        route_identity_id: str,
        source_key_set_id: str,
        target_key_set_id: str,
        source_owner_map_id: str,
        target_owner_map_id: str,
        source_value_id: str,
        target_value_id: str,
        source_content_root: str,
        target_content_root: str,
        transfer_id: str,
        target_entity_keys: ArrayLike,
        target_values: ArrayLike,
        successful: ArrayLike,
        /,
    ) -> None:
        from ..lifecycle import CompositionRole

        source_role = parse(source_role, CompositionRole, "source_role")

        identities = (
            migration_evidence_id,
            group_id,
            target_directory_id,
            source_entry_id,
            source_entry_record_id,
            source_structure_id,
            source_owner_id,
            source_revision_id,
            source_semantics_id,
            target_structure_id,
            key_identity_id,
            route_identity_id,
            source_key_set_id,
            target_key_set_id,
            source_owner_map_id,
            target_owner_map_id,
            source_value_id,
            target_value_id,
            source_content_root,
            target_content_root,
            transfer_id,
        )
        keys = np.asarray(target_entity_keys)
        target_array, target_payload, _, _ = _owner_local_state_payload(target_values)
        accepted = np.asarray(successful)
        if (
            any(not isinstance(value, str) or not value for value in identities)
            or source_key_set_id != target_key_set_id
            or isinstance(local_rank, bool)
            or not isinstance(local_rank, (int, np.integer))
            or isinstance(part_count, bool)
            or not isinstance(part_count, (int, np.integer))
            or part_count < 1
            or not 0 <= local_rank < part_count
            or keys.dtype.kind not in "iu"
            or keys.ndim != 2
            or keys.shape[1] < 1
            or (
                keys.dtype.kind == "u"
                and keys.size
                and int(np.max(keys)) > np.iinfo(np.int64).max
            )
            or (keys.size and np.any(keys < 0))
            or len({tuple(int(value) for value in row) for row in keys.tolist()})
            != keys.shape[0]
            or target_array.ndim == 0
            or target_array.shape[0] != keys.shape[0]
            or target_payload.ndim == 0
            or target_payload.shape[0] != keys.shape[0]
            or target_payload.dtype.hasobject
            or target_payload.dtype.kind not in "biufc"
            or not target_payload.dtype.isnative
            or accepted.shape != ()
            or accepted.dtype != np.dtype(np.bool_)
        ):
            raise ValueError("Owner-local keyed state transfer evidence is malformed.")
        self.migration_evidence_id = migration_evidence_id
        self.group_id = group_id
        self.local_rank = int(local_rank)
        self.part_count = int(part_count)
        self.target_directory_id = target_directory_id
        self.source_entry_id = source_entry_id
        self.source_entry_record_id = source_entry_record_id
        self.source_structure_id = source_structure_id
        self.source_owner_id = source_owner_id
        self.source_role = source_role
        self.source_revision_id = source_revision_id
        self.source_semantics_id = source_semantics_id
        self.target_structure_id = target_structure_id
        self.key_identity_id = key_identity_id
        self.route_identity_id = route_identity_id
        self.source_key_set_id = source_key_set_id
        self.target_key_set_id = target_key_set_id
        self.source_owner_map_id = source_owner_map_id
        self.target_owner_map_id = target_owner_map_id
        self.source_value_id = source_value_id
        self.target_value_id = target_value_id
        self.source_content_root = source_content_root
        self.target_content_root = target_content_root
        self.transfer_id = transfer_id
        self.target_entity_keys = jnp.asarray(keys, dtype=jnp.int64)
        self.target_values = jnp.asarray(target_array)
        self.successful = jnp.asarray(accepted, dtype=jnp.bool_)

    def composition_transport(
        self,
        source: CompositionEntry,
        target: CompositionEntry,
        /,
        *,
        target_revision_id: str | None = None,
    ) -> CompositionTransport:
        from ..lifecycle import CompositionEntry, CompositionTransport

        if not isinstance(source, CompositionEntry) or not isinstance(
            target, CompositionEntry
        ):
            raise TypeError("Graph state transport requires CompositionEntry values.")
        expected_target_revision = (
            self.source_revision_id if target_revision_id is None else target_revision_id
        )
        if not isinstance(expected_target_revision, str) or not expected_target_revision:
            raise ValueError("Graph state transport target revision must be non-empty.")
        if (
            source.entry_id != self.source_entry_id
            or source.record_id != self.source_entry_record_id
            or source.structure_id != self.source_structure_id
            or source.owner_id != self.source_owner_id
            or source.role != self.source_role
            or source.revision_id != self.source_revision_id
            or source.semantics_id != self.source_semantics_id
            or target.entry_id != self.source_entry_id
            or target.structure_id != self.target_structure_id
            or target.owner_id != self.source_owner_id
            or target.role != self.source_role
            or target.revision_id != expected_target_revision
            or target.semantics_id != self.source_semantics_id
            or _owner_local_state_value_identity(source.value) != self.source_value_id
            or _owner_local_state_value_identity(target.value) != self.target_value_id
        ):
            raise ValueError(
                "Graph state transport entries differ from their exact rows."
            )
        return CompositionTransport(
            "ownership-migration",
            (source.entry_id,),
            (target,),
            source_structure_ids=(source.structure_id,),
            route_id=canonical_fingerprint(
                {
                    "kind": "owner-local-graph-keyed-state-transport",
                    "transfer": self.transfer_id,
                    "target_entry": target.record_id,
                    "target_revision": target.revision_id,
                }
            ),
            successful=self.successful,
            source_content_root=self.source_content_root,
            target_content_root=self.target_content_root,
        )

    def gather_target_local_values(
        self,
        target_local_keys: ArrayLike,
        execution_group: ExecutionGroup,
        /,
        *,
        axis_name: str = "owner_local_graph_target_values",
    ) -> Array:
        """Fetch target-owned rows for a local FE/overset key list.

        Requests are resolved by the target's sharded key directory and values
        return directly from their current owners. Repeated local keys are
        preserved in the returned row order.
        """
        import jax

        if not isinstance(execution_group, ExecutionGroup):
            raise TypeError("execution_group must be ExecutionGroup.")
        members = execution_group.spec.process_indices
        try:
            rank = members.index(jax.process_index())
        except ValueError as error:
            raise ValueError(
                "The current process is outside the graph migration group."
            ) from error
        local_error = None
        request_keys = np.empty((0, self.target_entity_keys.shape[1]), dtype=np.int64)
        target_array: Array | np.ndarray = self.target_values
        implementation_id: str | None = None
        implementation: object | None = None
        try:
            raw_keys = np.asarray(target_local_keys)
            if raw_keys.dtype.kind not in "iu" or raw_keys.ndim not in (1, 2):
                raise ValueError("target_local_keys must be fixed-width integer keys.")
            if raw_keys.ndim == 1:
                raw_keys = raw_keys.reshape((-1, 1))
            if (
                raw_keys.shape[1] < 1
                or (
                    raw_keys.dtype.kind == "u"
                    and raw_keys.size
                    and int(np.max(raw_keys)) > np.iinfo(np.int64).max
                )
                or (raw_keys.size and np.any(raw_keys < 0))
            ):
                raise ValueError("target_local_keys contain invalid entity keys.")
            request_keys = np.asarray(raw_keys, dtype=np.int64)
            target_array, storage_values, implementation_id, implementation = (
                _owner_local_state_payload(self.target_values)
            )
            if (
                execution_group.spec.group_id != self.group_id
                or len(members) != self.part_count
                or rank != self.local_rank
                or not isinstance(axis_name, str)
                or not axis_name.strip()
                or request_keys.shape[1] != self.target_entity_keys.shape[1]
                or not bool(np.asarray(self.successful))
                or _owner_local_state_value_identity(target_array) != self.target_value_id
                or storage_values.ndim == 0
                or storage_values.shape[0] != self.target_entity_keys.shape[0]
                or storage_values.dtype.hasobject
                or storage_values.dtype.kind not in "biufc"
                or not storage_values.dtype.isnative
            ):
                raise ValueError(
                    "Target-local keyed value request is stale or malformed."
                )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            f"{axis_name}_input",
            "target-local keyed values",
        )

        value_kind = "typed-prng-key" if implementation_id is not None else "array"
        storage_tail_shape = tuple(int(value) for value in storage_values.shape[1:])
        logical_tail_shape = tuple(int(value) for value in target_array.shape[1:])
        row_width = prod(storage_tail_shape) * storage_values.dtype.itemsize
        header_id = canonical_fingerprint(
            {
                "kind": "owner-local-target-local-keyed-values",
                "transfer": self.transfer_id,
                "target_directory": self.target_directory_id,
                "key_identity": self.key_identity_id,
                "key_width": request_keys.shape[1],
                "value_kind": value_kind,
                "value_implementation": implementation_id,
                "logical_tail_shape": logical_tail_shape,
                "storage_dtype": storage_values.dtype.str,
                "storage_tail_shape": storage_tail_shape,
            }
        )
        header_rows = _owner_group_allgather(
            np.frombuffer(bytes.fromhex(header_id), dtype=np.uint8),
            execution_group,
            f"{axis_name}_layout",
        )
        if any(
            np.asarray(row, dtype=np.uint8).tobytes().hex() != header_id
            for row in header_rows
        ):
            raise ValueError("Target-local keyed value declarations differ across ranks.")

        owned_keys = np.asarray(self.target_entity_keys, dtype=np.int64)
        owner_by_request = _owner_local_keyed_target_owners(
            request_keys,
            owned_keys,
            np.full((owned_keys.shape[0],), rank, dtype=np.int32),
            rank,
            execution_group,
            self.key_identity_id,
            f"{axis_name}_directory",
            require_exact_coverage=False,
            allow_duplicate_queries=True,
        )
        owned_bytes = (
            np.ascontiguousarray(storage_values)
            .view(np.uint8)
            .reshape((owned_keys.shape[0], row_width))
            if row_width
            else np.zeros((owned_keys.shape[0], 0), dtype=np.uint8)
        )
        values_by_key = {
            tuple(int(value) for value in key): bytes(owned_bytes[row])
            for row, key in enumerate(owned_keys.tolist())
        }
        key_byte_width = 8 * owned_keys.shape[1]
        request_width = key_byte_width + 8
        outgoing_requests: list[list[np.ndarray]] = [[] for _ in range(self.part_count)]
        local_results: list[bytes | None] = [None] * request_keys.shape[0]
        local_error = None
        try:
            for row, key_row in enumerate(request_keys):
                key = tuple(int(value) for value in key_row)
                owner = int(owner_by_request[row])
                if owner == rank:
                    try:
                        local_results[row] = values_by_key[key]
                    except KeyError as error:
                        raise ValueError(
                            "Local target owner lacks a requested keyed value."
                        ) from error
                else:
                    key_bytes = np.asarray(key, dtype="<i8").view(np.uint8)
                    row_bytes = np.asarray([row], dtype="<i8").view(np.uint8)
                    outgoing_requests[owner].append(
                        np.concatenate((key_bytes, row_bytes))
                    )
            request_banks = tuple(
                np.stack(rows) if rows else np.zeros((0, request_width), dtype=np.uint8)
                for rows in outgoing_requests
            )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
            request_banks = tuple(
                np.zeros((0, request_width), dtype=np.uint8)
                for _ in range(self.part_count)
            )
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            f"{axis_name}_request_encode",
            "target-local keyed value requests",
        )
        incoming_requests = _owner_exchange_message_banks(
            request_banks, rank, execution_group, f"{axis_name}_requests"
        )
        replies: list[list[np.ndarray]] = [[] for _ in range(self.part_count)]
        local_error = None
        try:
            for requester, messages in incoming_requests.items():
                bank = np.asarray(messages, dtype=np.uint8)
                if bank.ndim != 2 or bank.shape[1] != request_width:
                    raise ValueError("Target-local value request layout is invalid.")
                for message in bank:
                    key = tuple(
                        int(value)
                        for value in np.frombuffer(
                            message[:key_byte_width].tobytes(), dtype="<i8"
                        )
                    )
                    row = int(
                        np.frombuffer(message[key_byte_width:].tobytes(), dtype="<i8")[0]
                    )
                    if row < 0 or key not in values_by_key:
                        raise ValueError(
                            "A target value owner lacks a requested entity key."
                        )
                    row_bytes = np.asarray([row], dtype="<i8").view(np.uint8)
                    value_bytes = np.frombuffer(values_by_key[key], dtype=np.uint8)
                    replies[int(requester)].append(
                        np.concatenate((row_bytes, value_bytes))
                    )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            f"{axis_name}_request_service",
            "target-local keyed value owner lookup",
        )
        reply_width = 8 + row_width
        reply_banks = tuple(
            np.stack(rows) if rows else np.zeros((0, reply_width), dtype=np.uint8)
            for rows in replies
        )
        incoming_replies = _owner_exchange_message_banks(
            reply_banks, rank, execution_group, f"{axis_name}_replies"
        )
        local_error = None
        try:
            for messages in incoming_replies.values():
                bank = np.asarray(messages, dtype=np.uint8)
                if bank.ndim != 2 or bank.shape[1] != reply_width:
                    raise ValueError("Target-local value reply layout is invalid.")
                for message in bank:
                    row = int(np.frombuffer(message[:8].tobytes(), dtype="<i8")[0])
                    if (
                        not 0 <= row < len(local_results)
                        or local_results[row] is not None
                    ):
                        raise ValueError(
                            "Target-local value replies are duplicated or invalid."
                        )
                    local_results[row] = bytes(message[8:])
            if any(value is None for value in local_results):
                raise ValueError(
                    "Target-local keyed value requests were not fully answered."
                )
            if row_width and local_results:
                result_bytes = np.stack(
                    [
                        np.frombuffer(value, dtype=np.uint8)
                        for value in local_results
                        if value is not None
                    ]
                )
                result_storage = (
                    result_bytes.reshape(-1)
                    .view(storage_values.dtype)
                    .reshape((request_keys.shape[0], *storage_tail_shape))
                )
            else:
                result_storage = np.empty(
                    (request_keys.shape[0], *storage_tail_shape),
                    dtype=storage_values.dtype,
                )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
            result_storage = np.empty(
                (request_keys.shape[0], *storage_tail_shape),
                dtype=storage_values.dtype,
            )
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            f"{axis_name}_reply_decode",
            "target-local keyed value coverage",
        )
        if implementation is not None:
            return jax.random.wrap_key_data(
                jnp.asarray(result_storage), impl=implementation
            )
        return jnp.asarray(result_storage)


def _transfer_owner_local_graph_keyed_values(
    evidence: OwnerLocalGraphMigrationEvidence,
    source_entry: CompositionEntry,
    source_entity_keys: ArrayLike,
    source_owner_ranks: ArrayLike,
    source_values: ArrayLike,
    target_entity_keys: ArrayLike,
    target_owner_ranks: ArrayLike,
    execution_group: ExecutionGroup,
    /,
    *,
    target_structure_id: str,
    key_identity_id: str,
    axis_name: str,
    value_fiber_shape: tuple[int, ...] | None = None,
) -> OwnerLocalGraphKeyedStateTransfer:
    import jax

    from ..lifecycle import CompositionEntry

    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    members = execution_group.spec.process_indices
    try:
        rank = members.index(jax.process_index())
    except ValueError as error:
        raise ValueError(
            "The current process is outside the graph migration group."
        ) from error

    def key_rows(value: ArrayLike, name: str, /) -> np.ndarray:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iu" or raw.ndim not in (1, 2):
            raise ValueError(f"{name} must contain fixed-width integer entity keys.")
        if raw.ndim == 1:
            raw = raw.reshape((-1, 1))
        if (
            raw.shape[1] < 1
            or (
                raw.dtype.kind == "u"
                and raw.size
                and int(np.max(raw)) > np.iinfo(np.int64).max
            )
            or (raw.size and np.any(raw < 0))
        ):
            raise ValueError(f"{name} contains invalid integer entity keys.")
        return np.asarray(raw, dtype=np.int64)

    local_error = None
    source_keys = np.empty((0, 1), dtype=np.int64)
    target_keys = np.empty((0, 1), dtype=np.int64)
    source_owners = np.empty((0,), dtype=np.int32)
    target_owners = np.empty((0,), dtype=np.int32)
    source_value_array: Array | np.ndarray = np.empty((0,), dtype=np.float64)
    values = np.empty((0,), dtype=np.float64)
    value_implementation_id: str | None = None
    logical_tail_shape: tuple[int, ...] = ()
    value_implementation: object | None = None
    try:
        if not isinstance(evidence, OwnerLocalGraphMigrationEvidence):
            raise TypeError("evidence must be OwnerLocalGraphMigrationEvidence.")
        if not isinstance(source_entry, CompositionEntry):
            raise TypeError("source_entry must be CompositionEntry.")
        source_keys = key_rows(source_entity_keys, "source_entity_keys")
        target_keys = key_rows(target_entity_keys, "target_entity_keys")
        source_owners_raw = np.asarray(source_owner_ranks)
        target_owners_raw = np.asarray(target_owner_ranks)
        (
            source_value_array,
            values,
            value_implementation_id,
            value_implementation,
        ) = _owner_local_state_payload(source_values)
        if value_fiber_shape is None:
            logical_tail_shape = tuple(int(size) for size in source_value_array.shape[1:])
        else:
            if type(value_fiber_shape) is not tuple or any(
                isinstance(size, (bool, np.bool_))
                or not isinstance(size, (int, np.integer))
                or int(size) <= 0
                for size in value_fiber_shape
            ):
                raise ValueError("Declared keyed value fiber shape is malformed.")
            logical_tail_shape = tuple(int(size) for size in value_fiber_shape)
        if source_value_array.shape != (
            source_keys.shape[0],
            *logical_tail_shape,
        ):
            raise ValueError(
                "Keyed state rows differ from their declared logical value fiber."
            )
        if (
            execution_group.spec.group_id != evidence.group_id
            or len(members) != evidence.part_count
            or rank != evidence.local_rank
            or not isinstance(axis_name, str)
            or not axis_name.strip()
            or not isinstance(target_structure_id, str)
            or not target_structure_id.strip()
            or not isinstance(key_identity_id, str)
            or not key_identity_id.strip()
            or source_keys.shape[1] != target_keys.shape[1]
            or source_owners_raw.dtype.kind not in "iu"
            or target_owners_raw.dtype.kind not in "iu"
            or source_owners_raw.shape != (source_keys.shape[0],)
            or target_owners_raw.shape != (target_keys.shape[0],)
            or source_value_array.ndim == 0
            or source_value_array.shape[0] != source_keys.shape[0]
            or values.ndim == 0
            or values.shape[0] != source_keys.shape[0]
            or (
                value_implementation_id is None
                and tuple(int(size) for size in values.shape[1:]) != logical_tail_shape
            )
            or values.dtype.hasobject
            or values.dtype.kind not in "biufc"
            or not values.dtype.isnative
            or _owner_local_state_value_identity(source_entry.value)
            != _owner_local_state_value_identity(source_value_array)
            or not np.array_equal(
                source_owners_raw.astype(np.int64, copy=False),
                np.full(source_keys.shape[0], rank, dtype=np.int64),
            )
            or not np.array_equal(
                target_owners_raw.astype(np.int64, copy=False),
                np.full(target_keys.shape[0], rank, dtype=np.int64),
            )
            or len({tuple(int(value) for value in row) for row in source_keys.tolist()})
            != source_keys.shape[0]
            or len({tuple(int(value) for value in row) for row in target_keys.tolist()})
            != target_keys.shape[0]
        ):
            raise ValueError(
                "Keyed state rows do not match their owned source/target maps."
            )
        source_owners = source_owners_raw.astype(np.int32, copy=False)
        target_owners = target_owners_raw.astype(np.int32, copy=False)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_input",
        "owner-local keyed state input",
    )

    parts = len(members)
    key_width = source_keys.shape[1]
    storage_tail_shape = tuple(int(value) for value in values.shape[1:])
    logical_fiber_width = prod(logical_tail_shape) if logical_tail_shape else 1
    storage_fiber_width = prod(storage_tail_shape) if storage_tail_shape else 1
    row_width = (
        storage_fiber_width
        if value_implementation_id is not None
        else logical_fiber_width
    ) * values.dtype.itemsize
    value_kind = "typed-prng-key" if value_implementation_id is not None else "array"
    common_migration_id = canonical_fingerprint(
        {
            "source_part": evidence.source_part_id,
            "source_face_coverage": evidence.periodic_face_coverage_id,
            "source_directory": evidence.source_directory_id,
            "target_part": evidence.target_part_id,
            "target_directory": evidence.target_directory_id,
            "group": evidence.group_id,
            "graph_plan": evidence.result.plan_id,
        }
    )
    route_identity_id = canonical_fingerprint(
        {
            "kind": "owner-local-graph-keyed-state-route",
            "migration": common_migration_id,
            "source_entry": source_entry.record_id,
            "source_structure": source_entry.structure_id,
            "target_structure": target_structure_id,
            "key_identity": key_identity_id,
        }
    )
    layout_id = canonical_fingerprint(
        {
            "kind": "owner-local-graph-keyed-state-layout",
            "migration": common_migration_id,
            "source_entry": source_entry.record_id,
            "source_structure": source_entry.structure_id,
            "target_structure": target_structure_id,
            "source_owner": source_entry.owner_id,
            "role": source_entry.role,
            "revision": source_entry.revision_id,
            "semantics": source_entry.semantics_id,
            "key_identity": key_identity_id,
            "route_identity": route_identity_id,
            "key_width": key_width,
            "value_kind": value_kind,
            "value_implementation": value_implementation_id,
            "logical_tail_shape": logical_tail_shape,
            "storage_dtype": values.dtype.str,
            "storage_tail_shape": storage_tail_shape,
        }
    )
    layout_rows = _owner_group_allgather(
        np.frombuffer(bytes.fromhex(layout_id), dtype=np.uint8),
        execution_group,
        f"{axis_name}_layout",
    )
    if any(
        np.asarray(value, dtype=np.uint8).tobytes().hex() != layout_id
        for value in layout_rows
    ):
        raise ValueError("Owner-local keyed state declarations differ across ranks.")

    key_set_header = {
        "migration": common_migration_id,
        "key_identity": key_identity_id,
        "key_width": key_width,
    }
    source_key_records = tuple(
        (tuple(int(value) for value in row), None) for row in source_keys.tolist()
    )
    target_key_records = tuple(
        (tuple(int(value) for value in row), None) for row in target_keys.tolist()
    )
    source_key_set_id = _owner_local_multiset_identity(
        "owner-local-graph-keyed-row-key-set",
        key_set_header,
        source_key_records,
        source_entry.entry_id,
        source_entry.revision_id,
        execution_group,
        f"{axis_name}_source_key_set",
    )
    target_key_set_id = _owner_local_multiset_identity(
        "owner-local-graph-keyed-row-key-set",
        key_set_header,
        target_key_records,
        source_entry.entry_id,
        source_entry.revision_id,
        execution_group,
        f"{axis_name}_target_key_set",
    )
    local_error = (
        None
        if source_key_set_id == target_key_set_id
        else "Source and target keyed state maps do not have the same exact key set."
    )
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_key_set",
        "owner-local keyed state key bijection",
    )

    source_owner_records = tuple(
        (
            tuple(int(value) for value in row),
            int(owner),
        )
        for row, owner in zip(source_keys.tolist(), source_owners, strict=True)
    )
    target_owner_records = tuple(
        (
            tuple(int(value) for value in row),
            int(owner),
        )
        for row, owner in zip(target_keys.tolist(), target_owners, strict=True)
    )
    source_owner_map_id = _owner_local_multiset_identity(
        "owner-local-graph-keyed-row-owner-map",
        {**key_set_header, "side": "source"},
        source_owner_records,
        source_entry.entry_id,
        source_entry.revision_id,
        execution_group,
        f"{axis_name}_source_owners",
    )
    target_owner_map_id = _owner_local_multiset_identity(
        "owner-local-graph-keyed-row-owner-map",
        {**key_set_header, "side": "target"},
        target_owner_records,
        source_entry.entry_id,
        source_entry.revision_id,
        execution_group,
        f"{axis_name}_target_owners",
    )
    target_owner_by_source_row = _owner_local_keyed_target_owners(
        source_keys,
        target_keys,
        target_owners,
        rank,
        execution_group,
        key_identity_id,
        f"{axis_name}_directory",
    )

    values = np.ascontiguousarray(values)
    byte_rows = (
        values.view(np.uint8).reshape((source_keys.shape[0], row_width))
        if row_width
        else np.zeros((source_keys.shape[0], 0), dtype=np.uint8)
    )
    key_byte_width = 8 * key_width
    message_width = key_byte_width + row_width
    outgoing_rows: list[list[np.ndarray]] = [[] for _ in range(parts)]
    local_values_by_key: dict[tuple[int, ...], bytes] = {}
    local_error = None
    try:
        for row, key_row in enumerate(source_keys):
            key = tuple(int(value) for value in key_row)
            owner = int(target_owner_by_source_row[row])
            key_bytes = np.asarray(key, dtype="<i8").view(np.uint8)
            payload = np.concatenate((key_bytes, byte_rows[row]))
            if owner == rank:
                if key in local_values_by_key:
                    raise ValueError("Keyed state migration duplicated a local row.")
                local_values_by_key[key] = bytes(byte_rows[row])
            else:
                outgoing_rows[owner].append(payload)
        outgoing = tuple(
            np.stack(rows) if rows else np.zeros((0, message_width), dtype=np.uint8)
            for rows in outgoing_rows
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
        outgoing = tuple(
            np.zeros((0, message_width), dtype=np.uint8) for _ in range(parts)
        )
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_encode",
        "owner-local keyed state messages",
    )
    incoming = _owner_exchange_message_banks(
        outgoing, rank, execution_group, f"{axis_name}_values"
    )
    local_error = None
    try:
        for messages in incoming.values():
            bank = np.asarray(messages, dtype=np.uint8)
            if bank.ndim != 2 or bank.shape[1] != message_width:
                raise ValueError(
                    "Owner-local keyed state messages have the wrong layout."
                )
            for message in bank:
                key = tuple(
                    int(value)
                    for value in np.frombuffer(
                        message[:key_byte_width].tobytes(), dtype="<i8"
                    )
                )
                if key in local_values_by_key:
                    raise ValueError("Keyed state migration duplicated an entity row.")
                local_values_by_key[key] = bytes(message[key_byte_width:])
        expected_keys = tuple(
            tuple(int(value) for value in row) for row in target_keys.tolist()
        )
        if set(local_values_by_key) != set(expected_keys):
            raise ValueError("Keyed state migration did not cover target entity rows.")
        if row_width and expected_keys:
            target_bytes = np.stack(
                [
                    np.frombuffer(local_values_by_key[key], dtype=np.uint8)
                    for key in expected_keys
                ]
            )
            target_storage_values = (
                target_bytes.reshape(-1)
                .view(values.dtype)
                .reshape((target_keys.shape[0], *storage_tail_shape))
            )
        else:
            target_storage_values = np.empty(
                (target_keys.shape[0], *storage_tail_shape), dtype=values.dtype
            )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
        target_storage_values = np.empty(
            (target_keys.shape[0], *storage_tail_shape), dtype=values.dtype
        )
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_decode",
        "owner-local keyed state values",
    )

    target_storage_values = np.ascontiguousarray(target_storage_values)
    source_value_rows = (
        values.view(np.uint8).reshape((source_keys.shape[0], row_width))
        if row_width
        else np.zeros((source_keys.shape[0], 0), dtype=np.uint8)
    )
    target_value_rows = (
        target_storage_values.view(np.uint8).reshape((target_keys.shape[0], row_width))
        if row_width
        else np.zeros((target_keys.shape[0], 0), dtype=np.uint8)
    )
    source_records = tuple(
        (
            tuple(int(value) for value in key),
            bytes(source_value_rows[row]).hex(),
        )
        for row, key in enumerate(source_keys)
    )
    target_records = tuple(
        (
            tuple(int(value) for value in key),
            bytes(target_value_rows[row]).hex(),
        )
        for row, key in enumerate(target_keys)
    )
    content_header = {
        "migration": common_migration_id,
        "key_identity": key_identity_id,
        "route_identity": route_identity_id,
        "entry_record": source_entry.record_id,
        "owner": source_entry.owner_id,
        "role": source_entry.role,
        "revision": source_entry.revision_id,
        "semantics": source_entry.semantics_id,
        "value_kind": value_kind,
        "value_implementation": value_implementation_id,
        "logical_tail_shape": logical_tail_shape,
        "storage_dtype": values.dtype.str,
        "storage_tail_shape": storage_tail_shape,
    }
    source_content_root = _owner_local_multiset_identity(
        "owner-local-graph-keyed-state-content",
        content_header,
        source_records,
        source_entry.entry_id,
        source_entry.revision_id,
        execution_group,
        f"{axis_name}_source_content",
    )
    target_content_root = _owner_local_multiset_identity(
        "owner-local-graph-keyed-state-content",
        content_header,
        target_records,
        source_entry.entry_id,
        source_entry.revision_id,
        execution_group,
        f"{axis_name}_target_content",
    )
    target_value_array = (
        jax.random.wrap_key_data(
            jnp.asarray(target_storage_values), impl=value_implementation
        )
        if value_implementation is not None
        else jnp.asarray(target_storage_values)
    )
    source_value_id = _owner_local_state_value_identity(source_value_array)
    target_value_id = _owner_local_state_value_identity(target_value_array)
    transfer_id = canonical_fingerprint(
        {
            "kind": "owner-local-graph-keyed-state-transfer",
            "migration": common_migration_id,
            "entry": source_entry.record_id,
            "source_structure": source_entry.structure_id,
            "target_structure": target_structure_id,
            "key_identity": key_identity_id,
            "route_identity": route_identity_id,
            "source_key_set": source_key_set_id,
            "target_key_set": target_key_set_id,
            "source_owner_map": source_owner_map_id,
            "target_owner_map": target_owner_map_id,
            "source_content": source_content_root,
            "target_content": target_content_root,
        }
    )
    return OwnerLocalGraphKeyedStateTransfer(
        evidence.evidence_id,
        evidence.group_id,
        rank,
        parts,
        evidence.target_directory_id,
        source_entry.entry_id,
        source_entry.record_id,
        source_entry.structure_id,
        source_entry.owner_id,
        source_entry.role,
        source_entry.revision_id,
        source_entry.semantics_id,
        target_structure_id,
        key_identity_id,
        route_identity_id,
        source_key_set_id,
        target_key_set_id,
        source_owner_map_id,
        target_owner_map_id,
        source_value_id,
        target_value_id,
        source_content_root,
        target_content_root,
        transfer_id,
        target_keys,
        target_value_array,
        source_content_root == target_content_root,
    )


def partition_owner_local_graph(
    assembly: MeshAssembly,
    plan: GraphPartitionPlan,
    execution_group: ExecutionGroup,
    /,
    *,
    axis_name: str = "owner_local_graph",
) -> OwnerLocalGraphPartitionEvidence:
    """Partition an exact periodic local view without gathering its graph or mesh.

    The result is an owner-label/evidence record. Geometry migration remains a
    separate accepted-state transaction and is never implied by label production.
    """

    import jax

    from .._execution_runtime import ExecutionGroup
    from ..discretization._view_support import OwnerLocalCellGeometry
    from ..graph._distributed_partition import (
        DistributedWeightedCSRGraph,
        partition_graph_distributed,
    )
    from ._assembly import MeshAssembly

    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be the current ExecutionGroup.")
    if not isinstance(assembly, MeshAssembly):
        raise TypeError("assembly must be MeshAssembly.")
    if not isinstance(plan, GraphPartitionPlan):
        raise TypeError("plan must be GraphPartitionPlan.")
    if not isinstance(axis_name, str) or not axis_name.strip():
        raise ValueError("axis_name must be a non-empty string.")
    members = execution_group.spec.process_indices
    devices = execution_group.devices
    try:
        rank = members.index(jax.process_index())
    except ValueError as error:
        raise ValueError("The current process is outside the execution group.") from error
    if (
        not execution_group.is_member
        or tuple(device.process_index for device in devices) != members
        or not assembly.owner_local
        or assembly.local_rank != rank
        or len(assembly.parts) != 1
        or plan.part_count != len(members)
    ):
        raise ValueError(
            "Owner-local graph partition inputs do not match the current group."
        )
    view = assembly.parts[0].carrier
    if not isinstance(view, OwnerLocalCellGeometry):
        raise TypeError("Owner-local graph partition requires OwnerLocalCellGeometry.")
    if (
        view.owner_rank != rank
        or view.owner_directory_data.rank != rank
        or view.owner_directory_data.part_count != len(members)
        or view.periodic_topology_id is None
        or view.periodic_face_coverage_id is None
    ):
        raise ValueError(
            "Owner-local graph source lacks its exact periodic owner/facet proof."
        )
    view.require_owner_directory_current(
        execution_group,
        axis_name=f"{axis_name}_directory",
    )
    # No heterogeneous cell-work model is declared; balance one integer unit per owned cell.
    owned_cell_work_weights = np.ones(
        np.asarray(view.owned_cell_ids).shape, dtype=np.int64
    )
    graph = DistributedWeightedCSRGraph(
        view.cell_neighbor_offsets,
        view.cell_neighbor_ids,
        entity_ids=view.owned_cell_ids,
        owner_ranks=view.owned_cell_owner_ranks,
        neighbor_owner_ranks=view.cell_neighbor_owner_ranks,
        edge_weights=view.cell_neighbor_weights,
        vertex_weights=owned_cell_work_weights,
    )
    result = partition_graph_distributed(
        graph,
        plan,
        execution_group,
        axis_name=axis_name,
    )
    if (
        not np.array_equal(
            np.asarray(result.entity_ids, dtype=np.int64),
            np.asarray(view.owned_cell_ids, dtype=np.int64),
        )
        or np.asarray(result.owners).shape != np.asarray(view.owned_cell_ids).shape
        or result.plan_id != plan.plan_id
    ):
        raise ValueError("Distributed graph labels do not bind the local owner rows.")
    return OwnerLocalGraphPartitionEvidence(
        result,
        view.local_view_id,
        view.periodic_face_coverage_id,
        view.distribution_id,
    )


__all__ = [
    "OwnerLocalGraphPartitionEvidence",
    "OwnerLocalGraphMigrationEvidence",
    "OwnerLocalGraphKeyedStateTransfer",
    "partition_owner_local_graph",
    "ShardedOwnerDirectoryData",
    "repack_sharded_owner_directory",
    "MeshDistributionTransition",
    "MeshPartitionEvidence",
    "MeshPartitionKind",
    "MeshPartitionPolicy",
    "prepare_distribution_transition",
    "prepare_mesh_distribution",
]
