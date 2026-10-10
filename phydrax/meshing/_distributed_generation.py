#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native initial construction partitioned by authoritative constrained patches.

Only authored source facts are replicated. Target chart CDT/cavity work resides
on the patch owner; curves are constructed in their incident source closure and
reconciled by neighbor packets before any target is exposed. This is not an
unconstrained, spatially decomposed CDT or distributed PLC recovery algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from time import monotonic

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from numpy.typing import NDArray

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import MeshcoreError
from ._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
    SurfaceMeshingSpec,
)
from ._device_generation import (
    certify_device_surface_initial,
    DeviceGenerationEvidence,
    DeviceGenerationLayout,
    DeviceGenerationStatus,
)
from ._domain import (
    _curve_request,
    _selected_strata,
    compile_surface_domain,
    CompiledSurfaceDomain,
)
from ._surface_generation import _curve_vertices, generate_surface, SurfaceConstruction
from .providers._native_curve import _QueryLedger
from .providers._native_options import NativeSurfaceSchedule
from .providers._native_sources import NativeSurfaceSource


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
                "Initial surface generation requires typed compiled source and specification."
            )
        if self.compiled.specification_id != self.specification.specification_id:
            raise ValueError(
                "Compiled source controls do not belong to this specification."
            )
        if not isinstance(self.schedule, NativeSurfaceSchedule) or not isinstance(
            self.layout, DeviceGenerationLayout
        ):
            raise TypeError(
                "Initial generation requires native schedule and device capacity layout."
            )
        if not isinstance(self.patch_owners, np.ndarray):
            raise TypeError("Patch owners must be an immutable host int32 vector.")
        if (
            self.patch_owners.dtype != np.int32
            or self.patch_owners.shape != self.compiled.patches.shape
            or self.patch_owners.flags.writeable
        ):
            raise ValueError(
                "Patch owners must be immutable int32 rows aligned with authoritative selected patches."
            )
        if (
            isinstance(self.partition_count, bool)
            or not isinstance(self.partition_count, int)
            or self.partition_count != jax.process_count()
        ):
            raise ValueError(
                "Initial preparation belongs to another actual process group."
            )
        if np.any(self.patch_owners < 0) or np.any(
            self.patch_owners >= self.partition_count
        ):
            raise ValueError("An authored patch owner is outside the process set.")
        if (
            isinstance(self.maximum_metadata_bytes, bool)
            or not isinstance(self.maximum_metadata_bytes, int)
            or self.maximum_metadata_bytes <= 0
        ):
            raise ValueError("maximum_metadata_bytes must be a positive integer.")
        required = self.partition_count * (
            8 * (self.compiled.curves.size + 2 * self.compiled.patches.size + 3) + 32
        )
        if required > self.maximum_metadata_bytes:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Initial source-stratum metadata exceeds its declared budget.",
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


def prepare_distributed_surface_generation(
    source: NativeSurfaceSource,
    specification: SurfaceMeshingSpec,
    schedule: NativeSurfaceSchedule,
    layout: DeviceGenerationLayout,
    patch_owners: NDArray[np.int32],
    /,
    *,
    maximum_metadata_bytes: int,
) -> PreparedDistributedSurfaceGeneration:
    """Bind ownership to authored patch identities before target generation."""
    if not isinstance(source, NativeSurfaceSource):
        raise TypeError(
            "Distributed initial preparation requires its actual native surface source."
        )
    compiled = compile_surface_domain(source.domain, specification)
    if not isinstance(patch_owners, np.ndarray):
        raise TypeError("Patch owners must be a host int32 vector.")
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


def _consensus(failure: MeshingFailure | None, /) -> None:
    categories = tuple(MeshingFailureCategory)
    code = 0 if failure is None else categories.index(failure.category) + 1
    rejected = np.asarray(multihost_utils.process_allgather(np.int32(code), tiled=False))
    rejected = rejected[rejected > 0]
    if rejected.size:
        category = categories[int(np.min(rejected)) - 1]
        if failure is not None and failure.category is category:
            raise failure
        raise MeshingFailure(
            category,
            "An owner rejected initial native construction; no generated part was published.",
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
    request = _curve_request(
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


def _neighbor_curve_packets(
    prepared: PreparedDistributedSurfaceGeneration,
    local: CompiledSurfaceDomain | None,
    boundary: SurfaceBoundaryData,
    /,
) -> tuple[SurfaceBoundaryData, NDArray[np.int64], int, int]:
    """Use the stable curve owner's native boundary before any target CDT."""
    compiled = prepared.compiled
    vertices, parameters, identifiers, corner_ids = boundary
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
        # One authoritative boundary packet reaches every incident owner.
        pairs.update((owners[0], other) for other in owners[1:])
    if not pairs:
        return boundary, np.asarray(shared, dtype=np.int64), 0, 0
    retained_boundary_bytes = (
        vertices.nbytes
        + sum(values.nbytes for values in parameters.values())
        + sum(ids.nbytes for ids in identifiers.values())
    )
    _consensus(
        MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Neighbor boundary worksets exceed the declared scratch budget.",
            stage="distributed-initial-constraints",
        )
        if 6 * capacity * 5 * np.dtype(np.float64).itemsize + retained_boundary_bytes
        > prepared.specification.limits.maximum_scratch_bytes
        else None
    )
    packet = np.full((capacity, 5), -1.0, dtype=np.float64)
    cursor = 0
    for curve, values in sorted(parameters.items()):
        count = values.size
        if cursor + count > capacity:
            raise RuntimeError(
                "Native source boundary exceeds its pre-reserved packet capacity."
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

        @partial(
            jax.shard_map,
            mesh=sharding.mesh,
            in_specs=PartitionSpec("initial_owner"),
            out_specs=PartitionSpec("initial_owner"),
            check_vma=False,
        )
        def exchange(value: jax.Array) -> jax.Array:
            return jax.lax.ppermute(value, "initial_owner", permutation)

        received_global = jax.jit(exchange)(data)
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
                        "An authoritative source curve packet has no complete ordered parameter interval."
                    )
                selected[curve] = (rows[:, 1].copy(), rows[:, 2:].copy())
        if jax.process_index() in (first, second):
            transferred += 2 * packet.nbytes
    if local is None:
        return (
            boundary,
            np.asarray(shared, dtype=np.int64),
            sum(jax.process_index() in pair for pair in pairs),
            transferred,
        )
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
    return (
        agreed,
        np.asarray(shared, dtype=np.int64),
        sum(jax.process_index() in pair for pair in pairs),
        transferred,
    )


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
        curves, patch_corners = _selected_strata(
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
) -> tuple[
    CompiledSurfaceDomain | None, MeshingLimits, SurfaceBoundaryData, _QueryLedger
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
    ledger = _QueryLedger(limits)
    local_patches = compiled.patches[prepared.patch_owners == jax.process_index()]
    if not local_patches.size:
        return None, limits, (np.empty((0, 3), dtype=np.float64), {}, {}, {}), ledger
    curves, corners = _selected_strata(compiled.domain, local_patches)
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
    closure = _local_compiled(
        prepared, np.union1d(local_patches, closure_patches), curves, corners, limits
    )
    boundary = _curve_vertices(
        closure, prepared.schedule, ledger, monotonic(), maximum_normal_angle
    )
    if boundary[0].shape[0] > limits.maximum_vertices:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Native incident source boundaries exceed the owner vertex capacity before target CDT.",
            stage="distributed-initial-boundary",
        )
    return local, limits, boundary, ledger


def _validate_boundary(
    compiled: CompiledSurfaceDomain,
    boundary: SurfaceBoundaryData,
    ledger: _QueryLedger,
    /,
) -> None:
    """Bind received coordinates/ordering to exact authored curve evaluations."""
    from ._trace import MeshingStageKind

    points, parameters, identifiers, corners = boundary
    if (
        set(parameters) != set(compiled.curves.tolist())
        or set(identifiers) != set(parameters)
        or set(corners) != set(compiled.corners.tolist())
        or points.shape[0] > ledger.limits.maximum_vertices
    ):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Agreed source boundary does not fit its exact local stratum/capacity closure.",
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
            "Agreed corner coordinates do not preserve their authoritative source realization.",
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
                "Agreed curve packet changes authored parameter/end-corner incidence.",
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
                "Neighbor coordinates do not reproduce the authoritative native curve evaluation.",
                stage="distributed-initial-constraints",
            )
        if np.any(
            np.linalg.norm(np.diff(points[ids], axis=0), axis=1)
            > compiled.curve_sizes[row] * (1.0 + 1.0e-9)
        ):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Agreed shared boundary misses its original physical size bound.",
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
                    "Agreed shared boundary misses its original continuous source fidelity bound.",
                    stage="distributed-initial-constraints",
                )
    if sorted(used) != list(range(points.shape[0])):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Agreed curve interiors/corners overlap or leave unused local identities.",
            stage="distributed-initial-constraints",
        )


def execute_distributed_surface_initial(
    prepared: PreparedDistributedSurfaceGeneration,
    /,
    *,
    minimum_angle: float,
    maximum_normal_angle: float = np.inf,
) -> DistributedSurfaceConstruction:
    """Construct owner patches, reconcile shared constraints, then release rows.

    A result publisher must still consume the full independent global domain,
    embedding, source-association and organization proofs. These generated rows
    are deliberately not masqueraded as an accepted bisection epoch.
    """
    compiled, layout = prepared.compiled, prepared.layout
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
        raise ValueError("Processes disagree on authoritative initial-generation inputs.")
    local = None
    limits = prepared.specification.limits
    boundary: SurfaceBoundaryData = (np.empty((0, 3), dtype=np.float64), {}, {}, {})
    ledger = _QueryLedger(limits)
    failure = None
    try:
        local, limits, boundary, ledger = _prepare_boundary(
            prepared, maximum_normal_angle
        )
    except MeshingFailure as error:
        failure = error
    except (MeshcoreError, ValueError) as error:
        failure = MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            str(error),
            stage="distributed-initial-boundary",
        )
    _consensus(failure)
    boundary, shared, packets, neighbor_bytes = _neighbor_curve_packets(
        prepared, local, boundary
    )
    failure = None
    try:
        if local is not None:
            _validate_boundary(local, boundary, ledger)
    except MeshingFailure as error:
        failure = error
    _consensus(failure)
    construction = None
    evidence = None
    failure = None
    try:
        if local is not None:
            quality = prepared.specification.quality_target
            construction = generate_surface(
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
            )
            signs = np.asarray(
                [
                    (-1 if compiled.domain.patches[patch].reversed else 1)
                    for patch in construction.triangle_patches
                ],
                dtype=np.int32,
            )
            evidence = certify_device_surface_initial(
                layout, construction.vertices, construction.triangle_parameters, signs
            )
            if evidence.status is not DeviceGenerationStatus.COMPLETE:
                category = (
                    MeshingFailureCategory.COMPLIANCE_FAILED
                    if evidence.status is DeviceGenerationStatus.INVALID_GEOMETRY
                    else MeshingFailureCategory.RESOURCE_EXHAUSTED
                )
                raise MeshingFailure(
                    category,
                    "Initial device predicate/capacity barrier rejected native owner rows.",
                    stage="distributed-initial-device",
                )
    except MeshingFailure as error:
        failure = error
    except (MeshcoreError, ValueError) as error:
        failure = MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            str(error),
            stage="distributed-initial-construction",
        )
    _consensus(failure)
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
    _consensus(
        MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Collective initial construction exceeds the unchanged global request budgets.",
            stage="distributed-initial-resources",
        )
        if exhausted
        else None
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
