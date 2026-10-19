#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Construct initial surface rows partitioned by authored constrained patches.

Authored source data is replicated. Target chart CDT/cavity work runs on the
patch owner; incident owners reconcile source-curve packets before publishing
any target rows.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import assert_never

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from numpy.typing import NDArray

from phydrax.meshing._contracts import MeshingLimits, SurfaceMeshingSpec
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import MeshcoreError, MeshcoreStatus, NativeExecutionScope
from .._strict import StrictModule
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from ..typing import checked
from . import (
    _device_adaptation as device_adaptation_owner,
    _surface_generation as surface_generation_owner,
)
from ._collective_failure import collective_failure_barrier
from ._device_generation import (
    certify_device_surface_initial,
    DeviceGenerationEvidence,
    DeviceGenerationLayout,
    DeviceGenerationStatus,
)
from ._domain import (
    compile_surface_domain,
    CompiledSurfaceDomain,
    curve_request,
    selected_strata,
)
from ._surface_construction import SurfaceConstruction
from ._trace import MeshingStageKind
from .providers import (
    _native_curve as native_curve_owner,
    _native_sources as native_sources_owner,
)
from .providers._native_schedules import NativeSurfaceSchedule


type SurfaceBoundaryData = tuple[
    NDArray[np.float64],
    dict[int, NDArray[np.float64]],
    dict[int, NDArray[np.int64]],
    dict[int, int],
]


@dataclass(frozen=True, slots=True)
class PreparedDistributedSurfaceGeneration:
    compiled: CompiledSurfaceDomain
    specification: SurfaceMeshingSpec
    schedule: NativeSurfaceSchedule
    layout: DeviceGenerationLayout
    patch_owners: NDArray[np.int32]
    maximum_metadata_bytes: int
    partition_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.compiled, CompiledSurfaceDomain) or not isinstance(
            self.specification, SurfaceMeshingSpec
        ):
            raise TypeError(
                f"initial_generation.compiled must be CompiledSurfaceDomain with SurfaceMeshingSpec; got compiled={type(self.compiled).__name__}, specification={type(self.specification).__name__}"
            )
        if self.compiled.specification_id != self.specification.specification_id:
            raise ValueError(
                f"initial_generation.compiled.specification_id must match specification.specification_id; got {self.compiled.specification_id!r} versus {self.specification.specification_id!r}"
            )
        if not isinstance(self.schedule, NativeSurfaceSchedule) or not isinstance(
            self.layout, DeviceGenerationLayout
        ):
            raise TypeError(
                f"initial_generation.schedule must pair NativeSurfaceSchedule with DeviceGenerationLayout; got schedule={type(self.schedule).__name__}, layout={type(self.layout).__name__}"
            )
        if not isinstance(self.patch_owners, np.ndarray):
            raise TypeError(
                f"initial_generation.patch_owners must be a host int32 vector; got {type(self.patch_owners).__name__}"
            )
        if (
            self.patch_owners.dtype != np.int32
            or self.patch_owners.shape != self.compiled.patches.shape
            or self.patch_owners.flags.writeable
        ):
            raise ValueError(
                "initial_generation.patch_owners must be immutable int32 rows aligned with selected patches; got incompatible dtype, shape, or mutability"
            )
        if (
            isinstance(self.partition_count, bool)
            or not isinstance(self.partition_count, int)
            or self.partition_count != jax.process_count()
        ):
            raise ValueError(
                "initial_generation.partition_count must match this process group; got incompatible owner count"
            )
        if np.any(self.patch_owners < 0) or np.any(
            self.patch_owners >= self.partition_count
        ):
            raise ValueError(
                f"initial_generation.patch_owners must lie in [0, partition_count); got owners={self.patch_owners!r}, partition_count={self.partition_count}"
            )
        if (
            isinstance(self.maximum_metadata_bytes, bool)
            or not isinstance(self.maximum_metadata_bytes, int)
            or self.maximum_metadata_bytes <= 0
        ):
            raise ValueError(
                f"initial_generation.maximum_metadata_bytes must be a positive integer; got {self.maximum_metadata_bytes!r}"
            )
        required = self.partition_count * (
            8 * (self.compiled.curves.size + 2 * self.compiled.patches.size + 3) + 32
        )
        if required > self.maximum_metadata_bytes:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                f"initial_generation.maximum_metadata_bytes must cover source-stratum metadata; got required={required}, capacity={self.maximum_metadata_bytes}",
                stage="distributed-initial-prepare",
            )


@dataclass(frozen=True, slots=True)
class DistributedSurfaceConstruction:
    """Owner-local generated rows; no accepted-result or global-embedding claim.

    Metadata bytes count materialized binding/order/resource summary arrays;
    neighbor bytes count sent plus received fixed-capacity payloads. Neither is
    a measured native/XLA peak-memory or all-device construction claim.
    """

    construction: SurfaceConstruction | None
    vertex_ids: NDArray[np.int64]
    vertex_owners: NDArray[np.int32]
    cell_ids: NDArray[np.int64]
    global_vertex_count: int
    global_cell_count: int
    shared_curve_ids: NDArray[np.int64]
    neighbor_packet_count: int
    neighbor_transferred_bytes: int
    metadata_bytes: int
    device_evidence: DeviceGenerationEvidence | None
    source_id: str
    source_revision: str
    specification_id: str


@checked
def prepare_distributed_surface_generation(
    source: native_sources_owner.NativeSurfaceSource,
    specification: SurfaceMeshingSpec,
    schedule: NativeSurfaceSchedule,
    layout: DeviceGenerationLayout,
    patch_owners: NDArray[np.int32],
    /,
    *,
    maximum_metadata_bytes: int,
) -> PreparedDistributedSurfaceGeneration:
    """Bind ownership to authored patch identities before target generation."""
    compiled = compile_surface_domain(source.domain, specification)
    if not isinstance(patch_owners, np.ndarray):
        raise TypeError(
            f"initial_generation.patch_owners must be a host int32 vector; got {type(patch_owners).__name__}"
        )
    owners = patch_owners.copy()
    owners.flags.writeable = False
    return PreparedDistributedSurfaceGeneration(
        compiled,
        specification,
        schedule,
        layout,
        owners,
        maximum_metadata_bytes,
        jax.process_count(),
    )


@contextmanager
def _initial_phase(stage: str, budget: CoordinateEnclosureBudget, /) -> Iterator[None]:
    failure = None
    try:
        with budget.measurement():
            yield
    except MeshingFailure as error:
        failure = error
    except (MeshcoreError, ValueError) as error:
        failure = initial_owner_failure(error, stage)
    collective_failure_barrier(
        failure,
        message="An owner rejected initial native construction; no generated part was published.",
        stage="distributed-initial-commit",
    )


def _local_compiled(
    prepared: PreparedDistributedSurfaceGeneration,
    patches: NDArray[np.int64],
    curves: NDArray[np.int64],
    corners: NDArray[np.int64],
    limits: MeshingLimits,
    /,
) -> CompiledSurfaceDomain:
    original = prepared.compiled
    rows = np.searchsorted(original.patches, patches)
    curve_rows = np.searchsorted(original.curves, curves)
    specification = eqx.tree_at(
        lambda value: value.limits, prepared.specification, limits
    )
    request = curve_request(
        original.domain,
        specification,
        curves,
        original.curve_sizes[curve_rows],
        original.curve_deviations[curve_rows],
    )
    return CompiledSurfaceDomain(
        original.domain,
        patches,
        curves,
        corners,
        original.patch_sizes[rows],
        original.curve_sizes[curve_rows],
        original.patch_deviations[rows],
        original.curve_deviations[curve_rows],
        request,
        original.source_sizing,
        original.patch_normal_angles[rows],
        original.specification_id,
    )


class _NeighborCurvePlan(StrictModule):
    capacity: int = eqx.field(static=True)
    pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    owners: dict[int, tuple[int, ...]] = eqx.field(static=True)
    shared: NDArray[np.int64]


class _CurveBoundaryPacket(StrictModule):
    data: jax.Array
    sharding: NamedSharding = eqx.field(static=True)
    packet_bytes: int = eqx.field(static=True)


class _AgreedCurveRows(StrictModule):
    selected: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]
    transferred: int = eqx.field(static=True)


class _ReconciledBoundary(StrictModule):
    boundary: SurfaceBoundaryData


def _prepare_neighbor_curve_plan(
    prepared: PreparedDistributedSurfaceGeneration, /
) -> _NeighborCurvePlan:
    compiled = prepared.compiled
    capacities = np.asarray(
        multihost_utils.process_allgather(
            np.int64(prepared.layout.vertex_capacity), tiled=False
        )
    )
    capacity = int(np.max(capacities)) + 2 * compiled.curves.size
    pairs: set[tuple[int, int]] = set()
    curve_owners: dict[int, tuple[int, ...]] = {}
    shared: list[int] = []
    for curve in compiled.curves.tolist():
        uses = [
            patch
            for patch, _ in compiled.domain.curve_uses(curve)
            if patch in compiled.patches
        ]
        incident = set(
            prepared.patch_owners[np.searchsorted(compiled.patches, uses)].tolist()
        )
        master = int(prepared.patch_owners[np.searchsorted(compiled.patches, min(uses))])
        owners = (master, *sorted(incident - {master}))
        curve_owners[curve] = owners
        if len(owners) > 1 and jax.process_index() in owners:
            shared.append(curve)
        # The curve owner's boundary packet reaches every incident owner.
        pairs.update((owners[0], other) for other in owners[1:])
    return _NeighborCurvePlan(
        capacity, tuple(sorted(pairs)), curve_owners, np.asarray(shared, dtype=np.int64)
    )


def _prepare_curve_boundary_packet(
    prepared: PreparedDistributedSurfaceGeneration,
    plan: _NeighborCurvePlan,
    boundary: SurfaceBoundaryData,
    /,
) -> _CurveBoundaryPacket:
    compiled, capacity = prepared.compiled, plan.capacity
    vertices, parameters, identifiers, _ = boundary
    retained_boundary_bytes = (
        vertices.nbytes
        + sum(values.nbytes for values in parameters.values())
        + sum(ids.nbytes for ids in identifiers.values())
    )
    collective_failure_barrier(
        MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "initial_generation.neighbor_packets must fit maximum_scratch_bytes; got packet worksets and retained boundary exceeding the scratch allowance",
            stage="distributed-initial-constraints",
        )
        if 6 * capacity * 5 * np.dtype(np.float64).itemsize + retained_boundary_bytes
        > prepared.specification.limits.maximum_scratch_bytes
        else None,
        message="An owner rejected initial native construction; no generated part was published.",
        stage="distributed-initial-commit",
    )
    packet = np.full((capacity, 5), -1.0, dtype=np.float64)
    cursor = 0
    for curve, values in sorted(parameters.items()):
        count = values.size
        if cursor + count > capacity:
            raise RuntimeError(
                f"initial_generation.curve_packet.capacity must cover retained boundary rows; got required={cursor + count}, capacity={capacity}"
            )
        packet[cursor : cursor + count, 0] = np.searchsorted(compiled.curves, curve)
        packet[cursor : cursor + count, 1] = values
        packet[cursor : cursor + count, 2:] = vertices[identifiers[curve]]
        cursor += count
    devices = [
        next(device for device in jax.devices() if device.process_index == rank)
        for rank in range(jax.process_count())
    ]
    sharding = NamedSharding(
        Mesh(np.asarray(devices), ("initial_owner",)), PartitionSpec("initial_owner")
    )
    data = jax.make_array_from_process_local_data(
        sharding, packet[None], (jax.process_count(), *packet.shape)
    )
    return _CurveBoundaryPacket(data, sharding, packet.nbytes)


@eqx.filter_jit
def _exchange_curve_packet(
    data: jax.Array, sharding: NamedSharding, permutation: tuple[tuple[int, int], ...], /
) -> jax.Array:
    @partial(
        jax.shard_map,
        mesh=sharding.mesh,
        in_specs=PartitionSpec("initial_owner"),
        out_specs=PartitionSpec("initial_owner"),
        check_vma=False,
    )
    def exchange(value: jax.Array) -> jax.Array:
        return jax.lax.ppermute(value, "initial_owner", permutation)

    return exchange(data)


def _exchange_curve_boundary_packets(
    compiled: CompiledSurfaceDomain,
    plan: _NeighborCurvePlan,
    packet: _CurveBoundaryPacket,
    boundary: SurfaceBoundaryData,
    /,
) -> _AgreedCurveRows:
    vertices, parameters, identifiers, _ = boundary
    pairs, curve_owners = plan.pairs, plan.owners
    selected = {
        curve: (values, vertices[identifiers[curve]])
        for curve, values in parameters.items()
    }
    transferred = 0
    for first, second in sorted(pairs):
        permutation = tuple(
            (rank, second if rank == first else first if rank == second else rank)
            for rank in range(jax.process_count())
        )
        received_global = _exchange_curve_packet(
            packet.data, packet.sharding, permutation
        )
        received = np.asarray(jax.device_get(received_global.addressable_shards[0].data))[
            0
        ]
        if jax.process_index() == second:
            for curve in parameters:
                if curve_owners[curve][0] != first:
                    continue
                rows = received[received[:, 0] == np.searchsorted(compiled.curves, curve)]
                if (
                    rows.shape[0] < 2
                    or rows[0, 1] != 0.0
                    or rows[-1, 1] != 1.0
                    or np.any(np.diff(rows[:, 1]) <= 0)
                ):
                    raise RuntimeError(
                        "initial_generation.curve_packet must span an increasing parameter interval from zero to one; got missing endpoints or unordered parameters"
                    )
                selected[curve] = (rows[:, 1].copy(), rows[:, 2:].copy())
        if jax.process_index() in (first, second):
            transferred += 2 * packet.packet_bytes
    return _AgreedCurveRows(selected, transferred)


def _assemble_curve_boundary(
    compiled: CompiledSurfaceDomain,
    local: CompiledSurfaceDomain,
    agreement: _AgreedCurveRows,
    /,
) -> _ReconciledBoundary:
    selected = agreement.selected
    coordinates = [compiled.domain.corner_points[local.corners]]
    corner_ids = {int(corner): row for row, corner in enumerate(local.corners)}
    count = local.corners.size
    parameters, identifiers = {}, {}
    for curve in local.curves.tolist():
        values, points = selected[curve]
        interior = values.size - 2
        source = compiled.domain.curves[curve]
        parameters[curve] = values
        identifiers[curve] = np.concatenate(
            (
                np.asarray((corner_ids[source.start],), dtype=np.int64),
                count + np.arange(interior, dtype=np.int64),
                np.asarray((corner_ids[source.end],), dtype=np.int64),
            )
        )
        coordinates.append(points[1:-1])
        count += interior
    agreed = (np.concatenate(coordinates), parameters, identifiers, corner_ids)
    return _ReconciledBoundary(agreed)


def _neighbor_curve_packets(
    prepared: PreparedDistributedSurfaceGeneration,
    local: CompiledSurfaceDomain | None,
    boundary: SurfaceBoundaryData,
    /,
) -> tuple[SurfaceBoundaryData, NDArray[np.int64], int, int]:
    """Use the stable curve owner's native boundary before any target CDT."""
    plan = _prepare_neighbor_curve_plan(prepared)
    if not plan.pairs:
        return boundary, plan.shared, 0, 0
    packet = _prepare_curve_boundary_packet(prepared, plan, boundary)
    agreement = _exchange_curve_boundary_packets(
        prepared.compiled, plan, packet, boundary
    )
    packets = sum(jax.process_index() in pair for pair in plan.pairs)
    if local is None:
        return boundary, plan.shared, packets, agreement.transferred
    agreed = _assemble_curve_boundary(prepared.compiled, local, agreement)
    return agreed.boundary, plan.shared, packets, agreement.transferred


def _canonical_ids(
    prepared: PreparedDistributedSurfaceGeneration,
    construction: SurfaceConstruction | None,
    counts: NDArray[np.int64],
    /,
) -> tuple[NDArray[np.int64], NDArray[np.int32], NDArray[np.int64], int, int]:
    compiled = prepared.compiled
    nc, npatch = compiled.curves.size, compiled.patches.size
    curve_counts = np.max(counts[:, :nc], axis=0)
    interior_counts = np.sum(counts[:, nc : nc + npatch], axis=0)
    cell_counts = np.sum(counts[:, nc + npatch :], axis=0)
    curve_offsets = compiled.corners.size + np.concatenate(
        (np.zeros((1,), dtype=np.int64), np.cumsum(curve_counts))
    )
    patch_offsets = curve_offsets[-1] + np.concatenate(
        (np.zeros((1,), dtype=np.int64), np.cumsum(interior_counts))
    )
    cell_offsets = np.concatenate(
        (np.zeros((1,), dtype=np.int64), np.cumsum(cell_counts))
    )
    vertex_count, cell_count = int(patch_offsets[-1]), int(cell_offsets[-1])
    if construction is None:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int64),
            vertex_count,
            cell_count,
        )
    ids = np.empty((construction.vertices.shape[0],), dtype=np.int64)
    owners = np.full(ids.shape, jax.process_index(), dtype=np.int32)
    dimensions, indices = (
        construction.vertex_source_dimensions,
        construction.vertex_source_indices,
    )
    corners = dimensions == 0
    ids[corners] = np.searchsorted(compiled.corners, indices[corners])
    for row, patch in enumerate(compiled.patches.tolist()):
        curves, patch_corners = selected_strata(
            compiled.domain, np.asarray((patch,), dtype=np.int64)
        )
        incident = corners & np.isin(indices, patch_corners)
        owners[incident] = np.minimum(owners[incident], prepared.patch_owners[row])
        for curve in curves.tolist():
            incident = (dimensions == 1) & (indices == curve)
            owners[incident] = np.minimum(owners[incident], prepared.patch_owners[row])
    for row, curve in enumerate(compiled.curves.tolist()):
        selected = np.flatnonzero((dimensions == 1) & (indices == curve))
        ids[selected] = curve_offsets[row] + np.arange(selected.size, dtype=np.int64)
    cells = np.empty(construction.triangle_patches.shape, dtype=np.int64)
    for row, patch in enumerate(compiled.patches.tolist()):
        selected = np.flatnonzero((dimensions == 2) & (indices == patch))
        ids[selected] = patch_offsets[row] + np.arange(selected.size, dtype=np.int64)
        selected_cells = np.flatnonzero(construction.triangle_patches == patch)
        cells[selected_cells] = cell_offsets[row] + np.arange(
            selected_cells.size, dtype=np.int64
        )
    return ids, owners, cells, vertex_count, cell_count


def _prepare_boundary(
    prepared: PreparedDistributedSurfaceGeneration,
    maximum_normal_angle: float,
    /,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> tuple[
    CompiledSurfaceDomain | None,
    MeshingLimits,
    SurfaceBoundaryData,
    native_curve_owner.QueryLedger,
]:
    compiled, layout = prepared.compiled, prepared.layout
    original = prepared.specification.limits
    limits = MeshingLimits(
        maximum_vertices=min(original.maximum_vertices, layout.vertex_capacity),
        maximum_edges=original.maximum_edges,
        maximum_faces=min(original.maximum_faces, layout.cell_capacity),
        maximum_cells=original.maximum_cells,
        maximum_connectivity_entries=original.maximum_connectivity_entries,
        maximum_data_bytes=original.maximum_data_bytes,
        maximum_work_units=min(original.maximum_work_units, layout.maximum_work_units),
        maximum_cavity_cells=original.maximum_cavity_cells,
        maximum_geometry_queries=original.maximum_geometry_queries,
        maximum_scratch_bytes=original.maximum_scratch_bytes,
        maximum_wall_seconds=original.maximum_wall_seconds,
    )
    ledger = native_curve_owner.QueryLedger(limits, budget=budget)
    local_patches = compiled.patches[prepared.patch_owners == jax.process_index()]
    if not local_patches.size:
        return None, limits, (np.empty((0, 3), dtype=np.float64), {}, {}, {}), ledger
    curves, corners = selected_strata(compiled.domain, local_patches)
    local = _local_compiled(prepared, local_patches, curves, corners, limits)
    closure_patches = np.unique(
        np.asarray(
            [
                patch
                for curve in curves.tolist()
                for patch, _ in compiled.domain.curve_uses(curve)
                if patch in compiled.patches
            ],
            dtype=np.int64,
        )
    )
    closure_patches = np.union1d(local_patches, closure_patches)
    closure_curves, closure_corners = selected_strata(compiled.domain, closure_patches)
    closure = _local_compiled(
        prepared, closure_patches, closure_curves, closure_corners, limits
    )
    boundary = surface_generation_owner.curve_vertices(
        closure,
        prepared.schedule,
        ledger,
        scope.operation_started,
        maximum_normal_angle,
        scope=scope,
        budget=budget,
    )
    if not np.array_equal(closure_curves, curves):
        points, parameters, identifiers, _ = boundary
        agreement = _AgreedCurveRows(
            {
                curve: (parameters[curve], points[identifiers[curve]])
                for curve in curves.tolist()
            },
            0,
        )
        boundary = _assemble_curve_boundary(compiled, local, agreement).boundary
    if boundary[0].shape[0] > limits.maximum_vertices:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            f"initial_generation.boundary.vertices must fit the owner vertex capacity before triangulation; got count={boundary[0].shape[0]}, capacity={limits.maximum_vertices}",
            stage="distributed-initial-boundary",
        )
    return local, limits, boundary, ledger


def _validate_boundary(
    compiled: CompiledSurfaceDomain,
    boundary: SurfaceBoundaryData,
    ledger: native_curve_owner.QueryLedger,
    /,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Bind received coordinates and ordering to authored curve evaluations."""

    points, parameters, identifiers, corners = boundary
    if (
        set(parameters) != set(compiled.curves.tolist())
        or set(identifiers) != set(parameters)
        or set(corners) != set(compiled.corners.tolist())
        or points.shape[0] > ledger.limits.maximum_vertices
    ):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "initial_generation.boundary must cover local curves and corners within vertex capacity; got mismatched stratum keys or excessive boundary vertices",
            stage="distributed-initial-constraints",
        )
    used = list(corners.values())
    corner_rows = np.asarray(
        [corners[int(corner)] for corner in compiled.corners], dtype=np.int64
    )
    if not np.array_equal(
        points[corner_rows], compiled.domain.corner_points[compiled.corners]
    ):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "initial_generation.boundary.corners must match authored corner coordinates; got different source realization",
            stage="distributed-initial-constraints",
        )
    for row, curve in enumerate(compiled.curves.tolist()):
        values, ids = parameters[curve], identifiers[curve]
        source = compiled.domain.curves[curve]
        if (
            values.size < 2
            or values[0] != 0.0
            or values[-1] != 1.0
            or np.any(np.diff(values) <= 0)
            or ids.shape != values.shape
            or ids[0] != corners[source.start]
            or ids[-1] != corners[source.end]
            or np.any(ids < 0)
            or np.any(ids >= points.shape[0])
        ):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "initial_generation.boundary.curve_packet must retain increasing parameters, valid vertex IDs, and authored endpoint incidence; got invalid interval or incidence",
                stage="distributed-initial-constraints",
            )
        used.extend(ids[1:-1].tolist())
        ledger.reserve(values.size - 2, 0, MeshingStageKind.CURVE_MESHING)
        exact_points = np.asarray(
            compiled.domain.curve_atlas.map(
                jnp.full((values.size - 2,), curve, dtype=jnp.int32),
                jnp.asarray(values[1:-1, None], dtype=jnp.float64),
            ),
            dtype=np.float64,
        ).reshape((-1, 3))
        if not np.array_equal(points[ids[1:-1]], exact_points):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "initial_generation.boundary.curve_coordinates must match native curve evaluation at received parameters; got different interior coordinates",
                stage="distributed-initial-constraints",
            )
        if np.any(
            np.linalg.norm(np.diff(points[ids], axis=0), axis=1)
            > compiled.curve_sizes[row] * (1.0 + 1.0e-9)
        ):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "initial_generation.boundary.curve_edges must satisfy the requested physical size bound; got an edge exceeding the bound",
                stage="distributed-initial-constraints",
            )
        if np.isfinite(compiled.curve_deviations[row]):
            ledger.reserve(
                values.size - 1, values.size - 1, MeshingStageKind.CURVE_MESHING
            )
            if np.any(
                compiled.domain.curve_interpolation_bounds(curve, values)
                > compiled.curve_deviations[row]
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    "initial_generation.boundary.curve_interpolation must satisfy the requested source-deviation bound; got an interpolation enclosure exceeding the bound",
                    stage="distributed-initial-constraints",
                )
    if sorted(used) != list(range(points.shape[0])):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "initial_generation.boundary.vertex_ids must cover local vertices once; got overlapping curve interiors or unused vertices",
            stage="distributed-initial-constraints",
        )


@checked
def execute_distributed_surface_initial(
    prepared: PreparedDistributedSurfaceGeneration,
    /,
    *,
    minimum_angle: float,
    maximum_normal_angle: float = np.inf,
) -> DistributedSurfaceConstruction:
    """Construct and collectively admit owner-local initial surface rows."""

    limits = prepared.specification.limits
    with (
        NativeExecutionScope(limits.native_limits()) as scope,
        device_adaptation_owner.device_coordinate_budget(limits, scope=scope) as budget,
    ):
        return _execute_distributed_surface_initial(
            prepared,
            minimum_angle=minimum_angle,
            maximum_normal_angle=maximum_normal_angle,
            scope=scope,
            budget=budget,
        )


def initial_owner_failure(
    error: MeshcoreError | ValueError, stage: str, /
) -> MeshingFailure:
    match error:
        case MeshcoreError():
            resource = error.status in (
                MeshcoreStatus.CAPACITY_EXCEEDED,
                MeshcoreStatus.TIMEOUT,
            )
        case CoordinateEnclosureResourceError():
            resource = True
        case ValueError():
            resource = False
        case value:
            assert_never(value)
    failure = MeshingFailure(
        MeshingFailureCategory.RESOURCE_EXHAUSTED
        if resource
        else MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
        str(error),
        stage=stage,
    )
    failure.__cause__ = error
    return failure


class _InitialOwnerConstruction(StrictModule):
    construction: SurfaceConstruction | None
    evidence: DeviceGenerationEvidence | None


def _construct_local_surface_initial(
    prepared: PreparedDistributedSurfaceGeneration,
    local: CompiledSurfaceDomain | None,
    boundary: SurfaceBoundaryData,
    ledger: native_curve_owner.QueryLedger,
    limits: MeshingLimits,
    minimum_angle: float,
    maximum_normal_angle: float,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    /,
) -> _InitialOwnerConstruction:
    if local is None:
        return _InitialOwnerConstruction(None, None)
    compiled, layout = prepared.compiled, prepared.layout
    quality = prepared.specification.quality_target
    construction = surface_generation_owner.generate_surface(
        local,
        prepared.schedule,
        limits,
        minimum_angle,
        maximum_normal_angle=maximum_normal_angle,
        prepared_boundary=boundary,
        boundary_queries=ledger.queries,
        boundary_work=ledger.work,
        required_minimum_angle=(
            quality.minimum_angle if quality is not None and quality.hard else 0.0
        ),
        size_compliance=prepared.specification.size_compliance,
        scope=scope,
        budget=budget,
    )
    signs = np.asarray(
        [
            (-1 if compiled.domain.patches[patch].reversed else 1)
            for patch in construction.triangle_patches
        ],
        dtype=np.int32,
    )
    evidence = certify_device_surface_initial(
        layout,
        construction.vertices,
        construction.triangle_parameters,
        signs,
        scope=scope,
    )
    if evidence.status is not DeviceGenerationStatus.COMPLETE:
        category = (
            MeshingFailureCategory.COMPLIANCE_FAILED
            if evidence.status is DeviceGenerationStatus.INVALID_GEOMETRY
            else MeshingFailureCategory.RESOURCE_EXHAUSTED
        )
        raise MeshingFailure(
            category,
            f"initial_generation.device_evidence.status must be COMPLETE; got {evidence.status}",
            stage="distributed-initial-device",
        )
    return _InitialOwnerConstruction(construction, evidence)


class _InitialInputBinding(StrictModule):
    bindings: NDArray[np.uint8]


def _bind_distributed_initial_inputs(
    prepared: PreparedDistributedSurfaceGeneration,
    minimum_angle: float,
    maximum_normal_angle: float,
    /,
) -> _InitialInputBinding:
    compiled = prepared.compiled
    binding = canonical_fingerprint(
        (
            compiled.compiled_id,
            array_tree_fingerprint(
                (
                    prepared.patch_owners,
                    np.asarray((minimum_angle, maximum_normal_angle), dtype=np.float64),
                ),
            ),
        )
    )
    bindings = np.asarray(
        multihost_utils.process_allgather(
            np.frombuffer(bytes.fromhex(binding), dtype=np.uint8), tiled=False
        )
    )
    if np.any(bindings != bindings[0]):
        raise ValueError(
            "initial_generation.inputs must agree across processes; got different source or control fingerprints"
        )
    return _InitialInputBinding(bindings)


def _execute_distributed_surface_initial(
    prepared: PreparedDistributedSurfaceGeneration,
    /,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    minimum_angle: float,
    maximum_normal_angle: float = np.inf,
) -> DistributedSurfaceConstruction:
    """Construct rows after shared-constraint and device admission.

    Publication additionally requires independent global domain, embedding,
    source-association, and organization proofs.
    """
    compiled = prepared.compiled
    binding = _bind_distributed_initial_inputs(
        prepared, minimum_angle, maximum_normal_angle
    )
    bindings = binding.bindings
    local = None
    limits = prepared.specification.limits
    boundary: SurfaceBoundaryData = (np.empty((0, 3), dtype=np.float64), {}, {}, {})
    ledger = native_curve_owner.QueryLedger(limits, budget=budget)
    with _initial_phase("distributed-initial-boundary", budget):
        local, limits, boundary, ledger = _prepare_boundary(
            prepared, maximum_normal_angle, scope=scope, budget=budget
        )
    boundary, shared, packets, neighbor_bytes = _neighbor_curve_packets(
        prepared, local, boundary
    )
    with _initial_phase("distributed-initial-constraints", budget):
        if local is not None:
            remaining = scope.remaining()
            ticket = scope.batch(
                min(
                    prepared.specification.limits.maximum_work_units - budget.work_units,
                    remaining.remaining_work_units,
                ),
                queries_upper=min(
                    prepared.specification.limits.maximum_geometry_queries,
                    remaining.remaining_geometry_queries,
                ),
                bytes_upper=min(
                    prepared.specification.limits.maximum_scratch_bytes,
                    remaining.remaining_managed_bytes,
                ),
                cavity_upper=min(
                    prepared.specification.limits.maximum_cavity_cells,
                    remaining.remaining_cavity_cells,
                ),
            )
            with budget.bind_ticket(ticket), ticket.native_target():
                _validate_boundary(local, boundary, ledger, scope=scope, budget=budget)
    owner = _InitialOwnerConstruction(None, None)
    with _initial_phase("distributed-initial-construction", budget):
        owner = _construct_local_surface_initial(
            prepared,
            local,
            boundary,
            ledger,
            limits,
            minimum_angle,
            maximum_normal_angle,
            scope,
            budget,
        )
    construction, evidence = owner.construction, owner.evidence
    local_counts = np.zeros(
        (compiled.curves.size + 2 * compiled.patches.size,), dtype=np.int64
    )
    if construction is not None:
        for curve, parameters in construction.curve_parameters:
            local_counts[np.searchsorted(compiled.curves, curve)] = parameters.size - 2
        for row, patch in enumerate(compiled.patches.tolist()):
            local_counts[compiled.curves.size + row] = np.sum(
                (construction.vertex_source_dimensions == 2)
                & (construction.vertex_source_indices == patch)
            )
            local_counts[compiled.curves.size + compiled.patches.size + row] = np.sum(
                construction.triangle_patches == patch
            )
    counts = np.asarray(multihost_utils.process_allgather(local_counts, tiled=False))
    ids, owners, cells, nv, nt = _canonical_ids(prepared, construction, counts)
    queries = (
        compiled.source_sizing.geometry_queries
        if construction is None
        else construction.geometry_queries
    )
    work = 0 if construction is None else construction.work_units
    if evidence is not None:
        work += evidence.device_candidates + evidence.exact_resolution_count
    resources = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((queries, work), dtype=np.int64), tiled=False
        )
    )
    limits = prepared.specification.limits
    exhausted = (
        nv > limits.maximum_vertices
        or nt > limits.maximum_faces
        or np.sum(resources[:, 0]) > limits.maximum_geometry_queries
        or np.sum(resources[:, 1]) > limits.maximum_work_units
    )
    collective_failure_barrier(
        MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "initial_generation.resources must fit global vertex, face, query, and work limits; got collective counts exceeding a request limit",
            stage="distributed-initial-resources",
        )
        if exhausted
        else None,
        message="An owner rejected initial native construction; no generated part was published.",
        stage="distributed-initial-commit",
    )
    return DistributedSurfaceConstruction(
        construction,
        ids,
        owners,
        cells,
        nv,
        nt,
        shared,
        packets,
        neighbor_bytes,
        bindings.nbytes + counts.nbytes + resources.nbytes,
        evidence,
        compiled.domain.source_id,
        compiled.domain.source_revision,
        compiled.specification_id,
    )
