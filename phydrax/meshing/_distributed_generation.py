#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Owner-local initial generation of authored MeshingDomain surface patches.

Patch interiors are triangulated only by their assigned owners. Shared source
curves use the owner of the lowest canonical incident patch as their boundary
sample steward; exact rational parameters are exchanged until every incident
owner has the same refined seam. Vertex and cell IDs are fingerprints of
source-stratum identity and exact chart parameters, so local parts retain stable
semantic identities without gathering a global mesh.
"""

from __future__ import annotations

import math
import struct
from fractions import Fraction
from itertools import combinations
from typing import final, Literal

import equinox as eqx
import jax
import numpy as np

import phydrax.geometry._meshing_domain as domain_owner
import phydrax.meshing.providers._native_surface as native_surface_owner
from phydrax._execution_runtime import ExecutionGroup
from phydrax._fingerprint import canonical_fingerprint
from phydrax._physical import SpatialCoordinateContract
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.graph import (
    GraphPartitionPlan,
    GraphPartitionResult,
    partition_graph,
    WeightedCSRGraph,
)
from phydrax.meshing._contracts import SurfaceMeshingSpec
from phydrax.meshing._distribution import (
    _owner_exchange_message_banks,
    _owner_group_allgather,
)
from phydrax.meshing._surface_construction import SurfaceConstruction
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.typing import checked


type _SourceEntityKey = tuple[str, tuple[str, ...]]
type _RationalIdentity = tuple[str, str]
type _SemanticVertexIdentity = (
    tuple[Literal["corner"], _SourceEntityKey]
    | tuple[Literal["curve"], _SourceEntityKey, _RationalIdentity]
    | tuple[
        Literal["patch"],
        _SourceEntityKey,
        _RationalIdentity,
        _RationalIdentity,
    ]
)


@final
class PreparedDistributedSurfaceGeneration(StrictModule, NonTrainableState):
    """Validated patch partition and collective context for surface generation."""

    surface: native_surface_owner._PreparedNativeSurface
    execution_group: ExecutionGroup = eqx.field(static=True)
    partition: GraphPartitionResult
    patch_owner_slots: np.ndarray
    patch_ids: np.ndarray
    patch_owner_ranks: np.ndarray
    curve_owners: tuple[tuple[int, tuple[int, ...]], ...] = eqx.field(static=True)
    curve_masters: tuple[tuple[int, int], ...] = eqx.field(static=True)
    corner_owners: tuple[tuple[int, int], ...] = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface: native_surface_owner._PreparedNativeSurface,
        execution_group: ExecutionGroup,
        partition: GraphPartitionResult,
        patch_owner_slots: np.ndarray,
        patch_ids: np.ndarray,
        patch_owner_ranks: np.ndarray,
        curve_owners: tuple[tuple[int, tuple[int, ...]], ...],
        curve_masters: tuple[tuple[int, int], ...],
        corner_owners: tuple[tuple[int, int], ...],
        axis_name: str,
        prepared_id: str,
        /,
    ) -> None:
        self.surface = surface
        self.execution_group = execution_group
        self.partition = partition
        self.patch_owner_slots = patch_owner_slots
        self.patch_ids = patch_ids
        self.patch_owner_ranks = patch_owner_ranks
        self.curve_owners = curve_owners
        self.curve_masters = curve_masters
        self.corner_owners = corner_owners
        self.axis_name = axis_name
        self.prepared_id = prepared_id


@final
class DistributedSurfaceConstruction(StrictModule, NonTrainableState):
    """One owner's rows and canonical IDs; not a globally assembled mesh.

    Shared-curve vertices repeat in each incident owner's local rows but carry
    the same canonical ID and source evaluation. Owner arrays contain execution
    group process indices; ``shared_curve_ids`` are canonical source-scope IDs.
    Cell IDs include their authored patch identity and unordered vertex IDs.
    ``construction`` is ``None`` for an owner with no assigned patches.
    """

    construction: SurfaceConstruction | None
    vertex_ids: tuple[str, ...] = eqx.field(static=True)
    vertex_owner_ranks: np.ndarray
    cell_ids: tuple[str, ...] = eqx.field(static=True)
    cell_owner_ranks: np.ndarray
    patch_ids: np.ndarray
    patch_owner_ranks: np.ndarray
    shared_curve_ids: np.ndarray
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        construction: SurfaceConstruction | None,
        vertex_ids: tuple[str, ...],
        vertex_owner_ranks: np.ndarray,
        cell_ids: tuple[str, ...],
        cell_owner_ranks: np.ndarray,
        patch_ids: np.ndarray,
        patch_owner_ranks: np.ndarray,
        shared_curve_ids: np.ndarray,
        source_id: str,
        source_revision: str,
        specification_id: str,
        prepared_id: str,
        /,
    ) -> None:
        self.construction = construction
        self.vertex_ids = vertex_ids
        self.vertex_owner_ranks = vertex_owner_ranks
        self.cell_ids = cell_ids
        self.cell_owner_ranks = cell_owner_ranks
        self.patch_ids = patch_ids
        self.patch_owner_ranks = patch_owner_ranks
        self.shared_curve_ids = shared_curve_ids
        self.source_id = source_id
        self.source_revision = source_revision
        self.specification_id = specification_id
        self.prepared_id = prepared_id


def _collective_error(
    local_error: Exception | None,
    execution_group: ExecutionGroup,
    axis_name: str,
    context: str,
    /,
) -> None:
    failed = np.asarray(
        _owner_group_allgather(
            np.asarray((local_error is not None,), dtype=np.bool_),
            execution_group,
            axis_name,
        ),
        dtype=np.bool_,
    ).reshape(-1)
    if np.any(failed):
        if local_error is not None:
            raise local_error
        raise ValueError(f"Another owner failed {context}.")


def _execution_rank(execution_group: ExecutionGroup, /) -> int:
    members = execution_group.spec.process_indices
    devices = execution_group.devices
    if (
        not execution_group.is_member
        or len(devices) != len(members)
        or tuple(device.process_index for device in devices) != members
    ):
        raise ValueError(
            "Distributed surface generation requires one ordered group device per member process."
        )
    try:
        return members.index(int(jax.process_index()))
    except ValueError as error:
        raise ValueError(
            "The current process is outside the distributed surface execution group."
        ) from error


def _patch_partition_graph(
    surface: native_surface_owner._PreparedNativeSurface,
    /,
) -> tuple[WeightedCSRGraph, tuple[int, ...]]:
    domain = surface.domain
    patches = surface.selected_patches
    patch_rows = {patch: row for row, patch in enumerate(patches)}
    curve_rows = native_surface_owner._trim_curves(domain, patches)
    pair_weights: dict[tuple[int, int], int] = {}
    selected = set(patches)
    for curve in curve_rows:
        incident = sorted(
            {patch for patch, _ in domain.curve_uses(curve) if patch in selected}
        )
        for first, second in combinations(incident, 2):
            edge = (patch_rows[first], patch_rows[second])
            pair_weights[edge] = pair_weights.get(edge, 0) + 1
    edges = np.asarray(tuple(pair_weights), dtype=np.int64).reshape((-1, 2))
    edge_weights = np.asarray(tuple(pair_weights.values()), dtype=np.int64)
    scope_ids, _ = native_surface_owner._scope_banks(domain)
    graph = WeightedCSRGraph.from_edges(
        len(patches),
        edges,
        edge_weights=edge_weights,
        entity_ids=scope_ids[2][np.asarray(patches, dtype=np.int64)],
    )
    return graph, curve_rows


def _patch_owner_maps(
    surface: native_surface_owner._PreparedNativeSurface,
    owner_slots: np.ndarray,
    curve_rows: tuple[int, ...],
    scope_ids: tuple[np.ndarray, ...],
    /,
) -> tuple[
    tuple[tuple[int, tuple[int, ...]], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]:
    domain = surface.domain
    patch_owners = dict(zip(surface.selected_patches, owner_slots.tolist(), strict=True))
    selected = set(surface.selected_patches)
    curve_owners: list[tuple[int, tuple[int, ...]]] = []
    curve_masters: list[tuple[int, int]] = []
    corner_owner_slots: dict[int, int] = {}
    for curve in curve_rows:
        incident_patches = tuple(
            sorted({patch for patch, _ in domain.curve_uses(curve) if patch in selected})
        )
        owners = tuple(sorted({patch_owners[patch] for patch in incident_patches}))
        if not owners:
            raise ValueError(
                "A selected source curve has no selected authored patch use."
            )
        master_patch = min(
            incident_patches,
            key=lambda patch: (int(scope_ids[2][patch]), patch_owners[patch]),
        )
        master = patch_owners[master_patch]
        curve_owners.append((curve, owners))
        curve_masters.append((curve, master))
        source_curve = domain.curves[curve]
        for patch in incident_patches:
            for corner in (source_curve.start, source_curve.end):
                previous = corner_owner_slots.get(corner)
                owner = patch_owners[patch]
                corner_owner_slots[corner] = (
                    owner if previous is None else min(previous, owner)
                )
    return (
        tuple(curve_owners),
        tuple(curve_masters),
        tuple(sorted(corner_owner_slots.items())),
    )


@checked
def prepare_distributed_surface_generation(
    source: domain_owner.MeshingDomain,
    specification: SurfaceMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    execution_group: ExecutionGroup,
    /,
    *,
    partition_plan: GraphPartitionPlan | None = None,
    axis_name: str = "distributed_surface",
) -> PreparedDistributedSurfaceGeneration:
    """Partition authored patches and prepare owner-local native surface work.

    The native surface provider owns source validation, chart construction,
    sizing, and root-cell ancestry. The native graph partitioner assigns patch
    rows, with shared authored curves weighted as patch adjacency.
    """
    axis = axis_name.strip() if isinstance(axis_name, str) else axis_name
    if not isinstance(axis, str) or not axis:
        raise ValueError("axis_name must be a nonempty string.")
    members = execution_group.spec.process_indices
    surface: native_surface_owner._PreparedNativeSurface | None = None
    partition: GraphPartitionResult | None = None
    patch_owner_slots = np.empty((0,), dtype=np.int32)
    patch_ids = np.empty((0,), dtype=np.int64)
    patch_owner_ranks = np.empty((0,), dtype=np.int32)
    curve_owners: tuple[tuple[int, tuple[int, ...]], ...] = ()
    curve_masters: tuple[tuple[int, int], ...] = ()
    corner_owners: tuple[tuple[int, int], ...] = ()
    partition_plan_id = ""
    local_error: Exception | None = None
    try:
        surface = native_surface_owner.prepare_native_surface(
            source,
            specification,
            options,
            coordinate_contract,
        )
        plan = partition_plan
        if plan is None:
            plan = GraphPartitionPlan(len(members), require_nonempty=False)
        partition_plan_id = plan.plan_id
        if plan.part_count != len(members):
            raise ValueError(
                "partition_plan.part_count must equal execution_group process count."
            )
        graph, curve_rows = _patch_partition_graph(surface)
        partition = partition_graph(graph, plan)
        patch_owner_slots = np.asarray(partition.owners, dtype=np.int32)
        if patch_owner_slots.shape != (len(surface.selected_patches),):
            raise ValueError("Native graph partition did not label every selected patch.")
        scope_ids, _ = native_surface_owner._scope_banks(source)
        selected_patches = np.asarray(surface.selected_patches, dtype=np.int64)
        patch_ids = scope_ids[2][selected_patches].copy()
        patch_owner_ranks = np.asarray(members, dtype=np.int32)[patch_owner_slots]
        curve_owners, curve_masters, corner_owners = _patch_owner_maps(
            surface,
            patch_owner_slots,
            curve_rows,
            scope_ids,
        )
        for values in (patch_owner_slots, patch_ids, patch_owner_ranks):
            values.setflags(write=False)
    except Exception as error:
        local_error = error
    _collective_error(
        local_error,
        execution_group,
        f"{axis}_prepare_failure",
        "distributed surface preparation",
    )
    if surface is None or partition is None:
        raise RuntimeError("Distributed surface preparation lost its validated inputs.")
    prepared_id = canonical_fingerprint(
        {
            "kind": "owner-local-meshing-domain-surface",
            "source": source.source_id,
            "revision": source.source_revision,
            "domain": source.domain_id,
            "coordinate_contract": coordinate_contract.spatial_id,
            "native_surface": surface.prepared_id,
            "partition": partition.result_id,
            "partition_plan": partition_plan_id,
            "group": execution_group.spec.group_id,
            "members": members,
            "axis": axis,
            "patch_owner_slots": patch_owner_slots.tolist(),
        }
    )
    identity = np.frombuffer(bytes.fromhex(prepared_id), dtype=np.uint8)
    identities = _owner_group_allgather(
        identity,
        execution_group,
        f"{axis}_prepared_identity",
    )
    if not np.all(identities == identities[0]):
        raise ValueError(
            "Distributed surface owners supplied different source, partition, or execution identities."
        )
    return PreparedDistributedSurfaceGeneration(
        surface,
        execution_group,
        partition,
        patch_owner_slots,
        patch_ids,
        patch_owner_ranks,
        curve_owners,
        curve_masters,
        corner_owners,
        axis,
        prepared_id,
    )


def _local_patches(
    prepared: PreparedDistributedSurfaceGeneration,
    local_patch_rows: tuple[int, ...],
    /,
) -> tuple[tuple[int, ...], np.ndarray]:
    surface = prepared.surface
    patches = tuple(surface.selected_patches[row] for row in local_patch_rows)
    sizes = np.asarray(
        [float(surface.size_values[row]) for row in local_patch_rows],
        dtype=np.float64,
    )
    return patches, sizes


def _initial_curve_parameters(
    prepared: PreparedDistributedSurfaceGeneration,
    local_patch_rows: tuple[int, ...],
    /,
) -> tuple[dict[int, set[Fraction]], native_surface_owner._PoleCharts]:
    """Serial initial seam samples restricted to the curves this owner trims.

    Curve chord sampling uses every selected incident patch size, so incident
    owners derive identical initial samples; pole-ring samples of remote patches
    arrive through the first steward reconciliation.
    """
    surface = prepared.surface
    patches, sizes = _local_patches(prepared, local_patch_rows)
    if not patches:
        return {}, {}
    local_curves = native_surface_owner._trim_curves(surface.domain, patches)
    parameters = native_surface_owner._curve_parameters(
        surface.domain,
        local_curves,
        surface.selected_patches,
        surface.size_values,
        surface.specification,
        surface.schedule,
    )
    pole_charts = native_surface_owner._pole_sampling(
        surface.domain,
        patches,
        sizes,
        surface.specification,
        surface.schedule,
        parameters,
    )
    native_surface_owner._integrate_pole_ring_parameters(
        surface.domain, patches, pole_charts, parameters
    )
    return parameters, pole_charts


def _generate_local_charts(
    prepared: PreparedDistributedSurfaceGeneration,
    local_patch_rows: tuple[int, ...],
    parameters_by_curve: dict[int, set[Fraction]],
    pole_charts: native_surface_owner._PoleCharts,
    /,
) -> list[native_surface_owner._ChartMesh]:
    surface = prepared.surface
    patches, _ = _local_patches(prepared, local_patch_rows)
    quality_angle = surface.schedule.quality_angle_degrees
    if surface.specification.quality_target is not None:
        quality_angle = max(
            quality_angle,
            math.degrees(surface.specification.quality_target.minimum_angle),
        )
    root_charts = native_surface_owner._surface_root_charts(surface.roots)
    charts = [
        native_surface_owner._make_chart_mesh(
            surface.domain,
            patch,
            parameters_by_curve,
            pole_charts,
            quality_angle,
        )
        for patch in patches
    ]
    for chart in charts:
        native_surface_owner._split_chart_at_source_cells(
            chart,
            root_charts.get(chart.patch, ()),
        )
    return charts


type _CurveSampleRecord = tuple[int, Fraction, tuple[float, float, float]]


def _canonical_curve_point(
    prepared: PreparedDistributedSurfaceGeneration,
    curve: int,
    parameter: Fraction,
    /,
) -> tuple[float, float, float]:
    domain = prepared.surface.domain
    source_curve = domain.curves[curve]
    if parameter == 0:
        point = np.asarray(domain.corner_points[source_curve.start], dtype=np.float64)
    elif parameter == 1:
        point = np.asarray(domain.corner_points[source_curve.end], dtype=np.float64)
    else:
        point = np.asarray(
            native_surface_owner._source_curve_point(domain, curve, parameter),
            dtype=np.float64,
        )
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise native_surface_owner._failure(
            "canonical source-curve evaluation is not one finite physical point",
            category=native_surface_owner.MeshingFailureCategory.INVALID_SOURCE,
            entity_ids=(int(domain.scope_entity_ids(1)[curve]),),
        )
    return float(point[0]), float(point[1]), float(point[2])


def _sample_records(
    records: tuple[_CurveSampleRecord, ...],
    width: int,
    /,
) -> np.ndarray:
    bank = np.zeros((len(records), width), dtype=np.uint8)
    for row, (curve, parameter, point) in enumerate(records):
        token = str(parameter).encode("ascii")
        encoded = (
            struct.pack("<qI", curve, len(token))
            + token
            + np.asarray(point, dtype="<f8").tobytes()
        )
        if len(encoded) > width:
            raise ValueError(
                "Exact curve sample exceeds the collectively prepared record width."
            )
        bank[row, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return bank


def _exchange_sample_records(
    records_by_owner: tuple[tuple[_CurveSampleRecord, ...], ...],
    prepared: PreparedDistributedSurfaceGeneration,
    phase: str,
    /,
) -> dict[int, np.ndarray]:
    local_width = max(
        (
            36 + len(str(parameter).encode("ascii"))
            for records in records_by_owner
            for _, parameter, _ in records
        ),
        default=0,
    )
    widths = np.asarray(
        _owner_group_allgather(
            np.asarray((local_width,), dtype=np.int64),
            prepared.execution_group,
            f"{prepared.axis_name}_{phase}_width",
        ),
        dtype=np.int64,
    )
    width = max(37, int(np.max(widths, initial=0)))
    banks: tuple[np.ndarray, ...] = ()
    local_error: Exception | None = None
    try:
        banks = tuple(_sample_records(records, width) for records in records_by_owner)
    except Exception as error:
        local_error = error
    _collective_error(
        local_error,
        prepared.execution_group,
        f"{prepared.axis_name}_{phase}_encoding_failure",
        "shared-curve sample encoding",
    )
    return _owner_exchange_message_banks(
        banks,
        _execution_rank(prepared.execution_group),
        prepared.execution_group,
        f"{prepared.axis_name}_{phase}",
    )


def _decode_sample_bank(bank: np.ndarray, /) -> tuple[_CurveSampleRecord, ...]:
    if bank.ndim != 2 or bank.dtype != np.dtype(np.uint8) or bank.shape[1] < 37:
        raise ValueError(
            "Received source-curve sample records have a malformed wire layout."
        )
    decoded: list[_CurveSampleRecord] = []
    for row in bank:
        curve, token_size = struct.unpack("<qI", row[:12].tobytes())
        token_stop = 12 + token_size
        point_stop = token_stop + 3 * np.dtype("<f8").itemsize
        if token_size == 0 or point_stop > bank.shape[1]:
            raise ValueError("Received source-curve sample record has invalid lengths.")
        parameter = Fraction(row[12:token_stop].tobytes().decode("ascii"))
        if not 0 <= parameter <= 1:
            raise ValueError(
                "Received source-curve sample lies outside its canonical parameter interval."
            )
        point_array = np.frombuffer(
            row[token_stop:point_stop].tobytes(),
            dtype="<f8",
        ).astype(np.float64, copy=False)
        if point_array.shape != (3,) or not np.all(np.isfinite(point_array)):
            raise ValueError("Received canonical source-curve coordinates are malformed.")
        point = (float(point_array[0]), float(point_array[1]), float(point_array[2]))
        decoded.append((curve, parameter, point))
    return tuple(decoded)


def _coordinates_match(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    /,
) -> bool:
    return (
        np.asarray(first, dtype=np.float64).tobytes()
        == np.asarray(second, dtype=np.float64).tobytes()
    )


def _validate_refined_shared_coordinates(
    prepared: PreparedDistributedSurfaceGeneration,
    charts: list[native_surface_owner._ChartMesh],
    refined: native_surface_owner._RefinedSurface | None,
    canonical_points: dict[int, dict[Fraction, tuple[float, float, float]]],
    /,
) -> None:
    if refined is None:
        return
    for chart, local_rows in zip(charts, refined.local_rows, strict=True):
        for (local_row, curve), parameter in chart.curve_parameters.items():
            points = canonical_points.get(curve)
            if points is None:
                continue
            canonical = points.get(parameter)
            if canonical is None:
                # Every split of a shared seam is exchanged in its refinement
                # round with its canonical point; a sample without one has no
                # incident owner agreement, so it is refused rather than published.
                domain = prepared.surface.domain
                raise native_surface_owner._failure(
                    "owner-local physical refinement split a shared source curve without incident owner agreement",
                    category=native_surface_owner.MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    entity_ids=(int(domain.scope_entity_ids(1)[curve]),),
                )
            physical_row = local_rows[local_row]
            physical = (
                float(refined.coordinates[physical_row, 0]),
                float(refined.coordinates[physical_row, 1]),
                float(refined.coordinates[physical_row, 2]),
            )
            if not _coordinates_match(physical, canonical):
                domain = prepared.surface.domain
                difference = np.linalg.norm(
                    np.asarray(physical, dtype=np.float64)
                    - np.asarray(canonical, dtype=np.float64)
                )
                allowance = max(
                    domain.tolerance * domain.scale,
                    64.0 * np.finfo(np.float64).eps * max(domain.scale, 1.0),
                )
                if difference > allowance:
                    raise native_surface_owner._failure(
                        "owner-local chart coordinates exceed the declared source-curve tolerance from the stewarded canonical point",
                        category=native_surface_owner.MeshingFailureCategory.INVALID_SOURCE,
                        entity_ids=(int(domain.scope_entity_ids(1)[curve]),),
                    )
                # Periodic chart gauges may differ within source accuracy or
                # binary64 representability; publish the steward's point as the
                # canonical seam coordinate before commit.
                if not refined.coordinates.flags.writeable:
                    raise ValueError(
                        "Reconciled surface coordinates must remain locally writable before commit."
                    )
                refined.coordinates[physical_row] = np.asarray(
                    canonical, dtype=np.float64
                )


@final
class _SeamRefinementRounds:
    """Lockstep owner-local refinement rounds that close shared trim splits.

    Serial refinement splits a trim edge jointly in every incident chart. Each
    round here first agrees on owner failures, then sends every requested split
    of a shared source curve, as its exact midpoint parameter and canonical
    point, to the curve's other incident owners. A receiver maps the midpoint to
    the trim interval of its own charts, so every incident chart splits the same
    interval in the same round, and the round continues while any owner still
    has schedule violations.
    """

    __slots__ = ("canonical_points", "charts", "prepared", "rank", "settled", "shared")

    def __init__(
        self,
        prepared: PreparedDistributedSurfaceGeneration,
        rank: int,
        charts: list[native_surface_owner._ChartMesh],
        canonical_points: dict[int, dict[Fraction, tuple[float, float, float]]],
        /,
    ) -> None:
        self.prepared = prepared
        self.rank = rank
        self.charts = charts
        self.canonical_points = canonical_points
        self.shared = {
            curve: owners for curve, owners in prepared.curve_owners if len(owners) > 1
        }
        # True once no further round can follow: refinement settled, or a
        # round failed collectively.
        self.settled = False

    def abandon(self) -> None:
        """Report this owner's failure to the round every other owner awaits."""
        self.settled = True
        _owner_group_allgather(
            np.asarray((True,), dtype=np.bool_),
            self.prepared.execution_group,
            f"{self.prepared.axis_name}_refinement_round_failure",
        )

    def __call__(
        self,
        splits: frozenset[native_surface_owner._TrimIdentity],
        pending: bool,
        /,
    ) -> tuple[frozenset[native_surface_owner._TrimIdentity], bool]:
        prepared = self.prepared
        group = prepared.execution_group
        axis = prepared.axis_name
        self.settled = True
        _collective_error(
            None,
            group,
            f"{axis}_refinement_round_failure",
            "owner-local physical refinement",
        )
        part_count = len(group.spec.process_indices)
        outgoing: list[list[_CurveSampleRecord]] = [[] for _ in range(part_count)]
        local_error: Exception | None = None
        try:
            for _, curve, lower, upper in sorted(splits):
                owners = self.shared.get(curve)
                if owners is None:
                    continue
                midpoint = (lower + upper) / 2
                point = _canonical_curve_point(prepared, curve, midpoint)
                self.canonical_points.setdefault(curve, {})[midpoint] = point
                for owner in owners:
                    if owner != self.rank:
                        outgoing[owner].append((curve, midpoint, point))
        except Exception as caught:
            local_error = caught
        _collective_error(
            local_error,
            group,
            f"{axis}_refinement_split_evaluation_failure",
            "shared trim split evaluation",
        )
        incoming = _exchange_sample_records(
            tuple(tuple(rows) for rows in outgoing),
            prepared,
            "refinement_trim_splits",
        )
        received: set[native_surface_owner._TrimIdentity] = set()
        local_error = None
        try:
            intervals: dict[tuple[int, Fraction], native_surface_owner._TrimIdentity] = {}
            for chart in self.charts:
                for curve, _, first, last in chart.boundary_edges.values():
                    lower, upper = min(first, last), max(first, last)
                    intervals[(curve, (lower + upper) / 2)] = (1, curve, lower, upper)
            for sender, bank in incoming.items():
                for curve, midpoint, point in _decode_sample_bank(bank):
                    owners = self.shared.get(curve)
                    if owners is None or sender not in owners or self.rank not in owners:
                        raise ValueError(
                            "A shared trim split arrived from a nonincident owner."
                        )
                    identity = intervals.get((curve, midpoint))
                    if identity is None:
                        raise native_surface_owner._failure(
                            "incident patch owners split different intervals of one shared source curve",
                            category=native_surface_owner.MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                            entity_ids=(
                                int(prepared.surface.domain.scope_entity_ids(1)[curve]),
                            ),
                        )
                    if not _coordinates_match(
                        _canonical_curve_point(prepared, curve, midpoint), point
                    ):
                        raise native_surface_owner._failure(
                            "an incident patch owner cannot reproduce a shared trim split's canonical source-curve coordinate",
                            category=native_surface_owner.MeshingFailureCategory.INVALID_SOURCE,
                            entity_ids=(
                                int(prepared.surface.domain.scope_entity_ids(1)[curve]),
                            ),
                        )
                    self.canonical_points.setdefault(curve, {})[midpoint] = point
                    received.add(identity)
        except Exception as caught:
            local_error = caught
        _collective_error(
            local_error,
            group,
            f"{axis}_refinement_split_receipt_failure",
            "shared trim split receipt",
        )
        any_pending = bool(
            np.any(
                np.asarray(
                    _owner_group_allgather(
                        np.asarray((pending,), dtype=np.bool_),
                        group,
                        f"{axis}_refinement_pending",
                    ),
                    dtype=np.bool_,
                )
            )
        )
        self.settled = not any_pending
        return frozenset(received), any_pending


def _reconcile_curve_parameters(
    prepared: PreparedDistributedSurfaceGeneration,
    rank: int,
    local_parameters: dict[int, set[Fraction]],
    /,
) -> tuple[
    dict[int, set[Fraction]],
    dict[int, dict[Fraction, tuple[float, float, float]]],
]:
    owners_by_curve = dict(prepared.curve_owners)
    masters_by_curve = dict(prepared.curve_masters)
    part_count = len(prepared.execution_group.spec.process_indices)
    shared = {curve: owners for curve, owners in prepared.curve_owners if len(owners) > 1}
    if not shared:
        return local_parameters, {}

    local_points: dict[int, dict[Fraction, tuple[float, float, float]]] = {}
    local_error: Exception | None = None
    try:
        local_points = {
            curve: {
                parameter: _canonical_curve_point(prepared, curve, parameter)
                for parameter in sorted(local_parameters[curve])
            }
            for curve in shared
            if rank in shared[curve]
        }
    except Exception as error:
        local_error = error
    _collective_error(
        local_error,
        prepared.execution_group,
        f"{prepared.axis_name}_curve_evaluation_failure",
        "canonical shared-curve evaluation",
    )

    outgoing_to_masters: list[list[_CurveSampleRecord]] = [[] for _ in range(part_count)]
    for curve, owners in shared.items():
        if rank not in owners or rank == masters_by_curve[curve]:
            continue
        outgoing_to_masters[masters_by_curve[curve]].extend(
            (curve, parameter, point)
            for parameter, point in sorted(local_points[curve].items())
        )
    incoming_to_masters = _exchange_sample_records(
        tuple(tuple(rows) for rows in outgoing_to_masters),
        prepared,
        "curve_samples_to_steward",
    )

    merged: dict[int, dict[Fraction, tuple[float, float, float]]] = {
        curve: dict(local_points[curve])
        for curve in shared
        if rank == masters_by_curve[curve]
    }
    local_error = None
    try:
        for sender, bank in incoming_to_masters.items():
            for curve, parameter, point in _decode_sample_bank(bank):
                owners = owners_by_curve.get(curve)
                if (
                    owners is None
                    or rank != masters_by_curve[curve]
                    or sender not in owners
                    or curve not in merged
                ):
                    raise ValueError(
                        "A source-curve sample arrived from a nonincident owner."
                    )
                previous = merged[curve].get(parameter)
                if previous is not None and not _coordinates_match(previous, point):
                    raise native_surface_owner._failure(
                        "incident patch owners cannot reconcile one exact source-curve sample coordinate",
                        category=native_surface_owner.MeshingFailureCategory.INVALID_SOURCE,
                        entity_ids=(
                            int(prepared.surface.domain.scope_entity_ids(1)[curve]),
                        ),
                    )
                merged[curve][parameter] = point
    except Exception as error:
        local_error = error
    _collective_error(
        local_error,
        prepared.execution_group,
        f"{prepared.axis_name}_curve_steward_failure",
        "shared-curve sample reconciliation",
    )

    outgoing_to_owners: list[list[_CurveSampleRecord]] = [[] for _ in range(part_count)]
    for curve, points in merged.items():
        for owner in shared[curve]:
            if owner == rank:
                continue
            outgoing_to_owners[owner].extend(
                (curve, parameter, point) for parameter, point in sorted(points.items())
            )
    incoming_to_owners = _exchange_sample_records(
        tuple(tuple(rows) for rows in outgoing_to_owners),
        prepared,
        "curve_samples_from_steward",
    )

    result = {curve: set(values) for curve, values in local_parameters.items()}
    canonical_points = {curve: dict(points) for curve, points in merged.items()}
    local_error = None
    try:
        for sender, bank in incoming_to_owners.items():
            for curve, parameter, point in _decode_sample_bank(bank):
                if (
                    curve not in shared
                    or sender != masters_by_curve[curve]
                    or rank not in shared[curve]
                ):
                    raise ValueError(
                        "A canonical seam sample arrived from a non-steward owner."
                    )
                expected = _canonical_curve_point(prepared, curve, parameter)
                if not _coordinates_match(expected, point):
                    raise native_surface_owner._failure(
                        "an incident patch owner cannot reproduce the steward's canonical source-curve coordinate",
                        category=native_surface_owner.MeshingFailureCategory.INVALID_SOURCE,
                        entity_ids=(
                            int(prepared.surface.domain.scope_entity_ids(1)[curve]),
                        ),
                    )
                result[curve].add(parameter)
                canonical_points.setdefault(curve, {})[parameter] = point
    except Exception as error:
        local_error = error
    _collective_error(
        local_error,
        prepared.execution_group,
        f"{prepared.axis_name}_curve_receiver_failure",
        "shared-curve sample receipt",
    )
    for curve, points in merged.items():
        result[curve] = set(points)
    return result, canonical_points


def _rational_identity(value: Fraction, /) -> tuple[str, str]:
    return str(value.numerator), str(value.denominator)


def _semantic_vertex_id(
    prepared: PreparedDistributedSurfaceGeneration,
    chart: native_surface_owner._ChartMesh,
    local_row: int,
    scope_ids: tuple[np.ndarray, ...],
    /,
) -> str:
    domain = prepared.surface.domain
    key = native_surface_owner._global_vertex_key(domain, chart, local_row)
    match key:
        case (0, corner):
            entity = domain.source_entity_key_for_scope(0, int(scope_ids[0][corner]))
            identity: _SemanticVertexIdentity = ("corner", entity)
        case (1, curve, parameter):
            entity = domain.source_entity_key_for_scope(1, int(scope_ids[1][curve]))
            identity = ("curve", entity, _rational_identity(parameter))
        case (2, patch, (u, v)):
            entity = domain.source_entity_key_for_scope(2, int(scope_ids[2][patch]))
            identity = (
                "patch",
                entity,
                _rational_identity(u),
                _rational_identity(v),
            )
        case _:
            raise ValueError(
                "Native surface chart returned an unknown semantic vertex identity."
            )
    return canonical_fingerprint(
        {
            "kind": "meshing-domain-surface-vertex",
            "source": domain.source_id,
            "revision": domain.source_revision,
            "identity": identity,
        }
    )


def _canonical_local_ids(
    prepared: PreparedDistributedSurfaceGeneration,
    construction: SurfaceConstruction,
    refined: native_surface_owner._RefinedSurface,
    /,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    scope_ids, _ = native_surface_owner._scope_banks(prepared.surface.domain)
    vertex_rows: list[str | None] = [None] * construction.vertices.shape[0]
    for chart, local_rows in zip(refined.charts, refined.local_rows, strict=True):
        for local_row in range(len(chart.points)):
            target_row = local_rows[local_row]
            identifier = _semantic_vertex_id(prepared, chart, local_row, scope_ids)
            existing = vertex_rows[target_row]
            if existing is not None and existing != identifier:
                raise ValueError(
                    "Local charts disagree on a canonical shared-vertex identity."
                )
            vertex_rows[target_row] = identifier
    if any(identifier is None for identifier in vertex_rows):
        raise ValueError(
            "Canonical surface identities do not cover every local construction vertex."
        )
    vertex_ids = tuple(identifier for identifier in vertex_rows if identifier is not None)
    if len(set(vertex_ids)) != len(vertex_ids):
        raise ValueError(
            "Distinct owner-local source vertices have a canonical identity collision."
        )
    patch_rows = np.asarray(construction.triangle_patches, dtype=np.int64)
    triangles = np.asarray(construction.triangles, dtype=np.int64)
    if triangles.shape != (patch_rows.size, 3):
        raise ValueError(
            "Owner-local surface cells do not align with their authored patches."
        )
    cell_ids = tuple(
        canonical_fingerprint(
            {
                "kind": "meshing-domain-surface-cell",
                "source": prepared.surface.domain.source_id,
                "revision": prepared.surface.domain.source_revision,
                "patch": prepared.surface.domain.source_entity_key_for_scope(
                    2, int(scope_ids[2][int(patch)])
                ),
                "vertices": tuple(sorted(vertex_ids[int(row)] for row in triangle)),
            }
        )
        for patch, triangle in zip(patch_rows.tolist(), triangles.tolist(), strict=True)
    )
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError(
            "Distinct owner-local surface cells have a canonical identity collision."
        )
    return vertex_ids, cell_ids


def _local_owner_arrays(
    prepared: PreparedDistributedSurfaceGeneration,
    construction: SurfaceConstruction | None,
    vertex_ids: tuple[str, ...],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    if construction is None:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
    patch_owner_slots = dict(
        zip(
            prepared.surface.selected_patches,
            prepared.patch_owner_slots.tolist(),
            strict=True,
        )
    )
    curve_masters = dict(prepared.curve_masters)
    corner_owners = dict(prepared.corner_owners)
    vertex_owners = np.empty((len(vertex_ids),), dtype=np.int32)
    for row, (dimension, source_index) in enumerate(
        zip(
            np.asarray(construction.vertex_source_dimensions).tolist(),
            np.asarray(construction.vertex_source_indices).tolist(),
            strict=True,
        )
    ):
        if dimension == 0:
            owner = corner_owners[int(source_index)]
        elif dimension == 1:
            owner = curve_masters[int(source_index)]
        elif dimension == 2:
            owner = patch_owner_slots[int(source_index)]
        else:
            raise ValueError(
                "Owner-local surface vertex has an unsupported source dimension."
            )
        vertex_owners[row] = prepared.execution_group.spec.process_indices[owner]
    cell_owners = np.asarray(
        [
            prepared.execution_group.spec.process_indices[patch_owner_slots[int(patch)]]
            for patch in np.asarray(
                construction.triangle_patches, dtype=np.int64
            ).tolist()
        ],
        dtype=np.int32,
    )
    return vertex_owners, cell_owners


@checked
def execute_distributed_surface_initial(
    prepared: PreparedDistributedSurfaceGeneration,
    /,
) -> DistributedSurfaceConstruction:
    """Generate owner-local patch charts and close every shared seam iteratively.

    This is the serial chart fixed point executed per owner: every round first
    unions the seam samples of all incident owners through the curve steward,
    then regenerates the local charts from that common sample set. Physical
    refinement then runs in lockstep rounds; a trim split requested by any
    incident owner is applied in every incident chart of the same round, so
    refinement reproduces the serial construction patch by patch.
    """
    rank = _execution_rank(prepared.execution_group)
    domain = prepared.surface.domain
    local_patch_rows: tuple[int, ...] = ()
    local_parameters: dict[int, set[Fraction]] = {}
    pole_charts: native_surface_owner._PoleCharts = {}
    local_error: Exception | None = None
    try:
        domain.require_current(domain.source_id, domain.source_revision)
        local_patch_rows = tuple(
            row
            for row, owner in enumerate(prepared.patch_owner_slots.tolist())
            if owner == rank
        )
        local_parameters, pole_charts = _initial_curve_parameters(
            prepared, local_patch_rows
        )
    except Exception as error:
        local_error = error
    _collective_error(
        local_error,
        prepared.execution_group,
        f"{prepared.axis_name}_execute_setup_failure",
        "distributed surface execution setup",
    )
    charts: list[native_surface_owner._ChartMesh] = []
    canonical_points: dict[int, dict[Fraction, tuple[float, float, float]]] = {}

    while True:
        local_error = None
        try:
            reconciled, canonical_points = _reconcile_curve_parameters(
                prepared,
                rank,
                local_parameters,
            )
            if any(
                not local_parameters[curve].issubset(reconciled[curve])
                for curve in local_parameters
            ):
                raise ValueError("Shared-curve sample refinement must be monotone.")
            local_parameters = reconciled
        except Exception as error:
            local_error = error
        _collective_error(
            local_error,
            prepared.execution_group,
            f"{prepared.axis_name}_seam_reconciliation_failure",
            "shared-seam reconciliation",
        )
        local_error = None
        discovered = local_parameters
        try:
            charts = _generate_local_charts(
                prepared,
                local_patch_rows,
                local_parameters,
                pole_charts,
            )
            discovered = {
                curve: set(values) for curve, values in local_parameters.items()
            }
            for chart in charts:
                for (local_row, curve), parameter in chart.curve_parameters.items():
                    if local_row < 0 or curve not in discovered:
                        raise ValueError(
                            "A local chart contains an unowned source-curve sample."
                        )
                    discovered[curve].add(parameter)
            for parameters in discovered.values():
                if not {Fraction(0), Fraction(1)}.issubset(parameters):
                    raise ValueError(
                        "A local source-curve boundary lost an authored endpoint."
                    )
        except Exception as error:
            local_error = error
        _collective_error(
            local_error,
            prepared.execution_group,
            f"{prepared.axis_name}_chart_failure",
            "owner-local chart generation",
        )
        changed = any(
            discovered[curve] != local_parameters[curve] for curve in local_parameters
        )
        changed_by_owner = np.asarray(
            _owner_group_allgather(
                np.asarray((changed,), dtype=np.bool_),
                prepared.execution_group,
                f"{prepared.axis_name}_seam_convergence",
            ),
            dtype=np.bool_,
        )
        if not np.any(changed_by_owner):
            break
        local_parameters = discovered

    refined: native_surface_owner._RefinedSurface | None = None
    rounds = _SeamRefinementRounds(prepared, rank, charts, canonical_points)
    local_error = None
    try:
        if charts:
            _, local_sizes = _local_patches(prepared, local_patch_rows)
            refined = native_surface_owner._refine_charts(
                domain,
                charts,
                local_sizes,
                prepared.surface.specification,
                prepared.surface.schedule,
                seam_rounds=rounds,
            )
        else:
            while rounds(frozenset(), False)[1]:
                pass
        _validate_refined_shared_coordinates(
            prepared,
            charts,
            refined,
            canonical_points,
        )
    except Exception as error:
        local_error = error
        if not rounds.settled:
            rounds.abandon()
    _collective_error(
        local_error,
        prepared.execution_group,
        f"{prepared.axis_name}_refinement_failure",
        "owner-local physical refinement",
    )

    construction: SurfaceConstruction | None = None
    vertex_ids: tuple[str, ...] = ()
    cell_ids: tuple[str, ...] = ()
    shared_curve_ids = np.empty((0,), dtype=np.int64)
    local_error = None
    try:
        if refined is not None:
            quality = prepared.surface.specification.quality_target
            if (
                quality is not None
                and quality.hard
                and refined.minimum_angle < math.degrees(quality.minimum_angle)
            ):
                raise native_surface_owner._failure(
                    "owner-local physical surface triangles miss the hard minimum-angle target",
                    category=native_surface_owner.MeshingFailureCategory.COMPLIANCE_FAILED,
                )
            curve_ids = native_surface_owner._trim_curves(
                domain,
                tuple(prepared.surface.selected_patches[row] for row in local_patch_rows),
            )
            construction = native_surface_owner._surface_construction(
                domain,
                refined,
                prepared.surface.roots,
                curve_ids,
                prepared.surface.schedule,
            )
            vertex_ids, cell_ids = _canonical_local_ids(
                prepared,
                construction,
                refined,
            )
        vertex_owner_ranks, cell_owner_ranks = _local_owner_arrays(
            prepared,
            construction,
            vertex_ids,
        )
        shared_curve_ids = np.asarray(
            [
                int(domain.scope_entity_ids(1)[curve])
                for curve, owners in prepared.curve_owners
                if len(owners) > 1
            ],
            dtype=np.int64,
        )
        shared_curve_ids.setflags(write=False)
    except Exception as error:
        local_error = error
        vertex_owner_ranks = np.empty((0,), dtype=np.int32)
        cell_owner_ranks = np.empty((0,), dtype=np.int32)
        shared_curve_ids = np.empty((0,), dtype=np.int64)
    _collective_error(
        local_error,
        prepared.execution_group,
        f"{prepared.axis_name}_commit_failure",
        "owner-local surface construction commit",
    )
    vertex_owner_ranks.setflags(write=False)
    cell_owner_ranks.setflags(write=False)
    return DistributedSurfaceConstruction(
        construction,
        vertex_ids,
        vertex_owner_ranks,
        cell_ids,
        cell_owner_ranks,
        prepared.patch_ids,
        prepared.patch_owner_ranks,
        shared_curve_ids,
        domain.source_id,
        domain.source_revision,
        prepared.surface.specification.specification_id,
        prepared.prepared_id,
    )


__all__ = [
    "DistributedSurfaceConstruction",
    "PreparedDistributedSurfaceGeneration",
    "execute_distributed_surface_initial",
    "prepare_distributed_surface_generation",
]
