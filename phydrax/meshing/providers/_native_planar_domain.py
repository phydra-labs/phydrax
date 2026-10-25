#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared planar size fields, constraints, and publication evidence."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np

import phydrax.discretization as cell_mesh_owner
import phydrax.geometry as triangulation_owner
from phydrax.geometry._planar_primitives import rational_points
from phydrax.meshing._association_state import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._contracts import MeshingLimits, SurfaceMeshingSpec
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._meshcore import (
    current_native_host_workspace,
    exact_orient2d,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionScope,
    NativeResourceFailure,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...discretization._exact_plc_geometry import ExactPlcCellGeometrySource
from ...geometry._domain_coverage import PiecewiseLinearDomain
from ...geometry._source_fidelity import fidelity_norm_bound
from ...typing import checked
from .._controls import FeatureKind
from .._organization_records import MeshLabel, MeshPatch
from .._result import MeshingComplianceReport
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import (
    resolve_size_controls,
    size_statistic_quantile,
    SizeCompliancePolicy,
    SizeControlStrength,
    SizeFieldDomain,
    UniformSizeControl,
)
from .._trace import MeshingStageKind
from . import _native_sources as sources_owner
from ._native_planar_construction import (
    lattice_spacing,
    uniform_construction_size,
    uniform_median_strategy,
)
from ._native_route_checks import check_deadline, unique_edges


# Point-by-segment distance entries evaluated per chunk of size-field queries.
_FIELD_WORKING_ENTRIES = 1 << 20
# Host bytes the CDT wrapper allocates per possible output point before the
# native build: the constraint-witness parent, segment and ratio rows.
_CDT_WITNESS_BYTES = 20
# Host bytes a CDT build may hold per output point after the native build:
# output coordinates and vertex map, the frozen copy, the input-point unique
# sort and the content fingerprints.
_CDT_POINT_BYTES = 140
# Host bytes per output triangle: cell and segment-id banks (48), their frozen
# copies, the triangle-vertex unique sort, the minimum-angle scan temporaries
# and the content fingerprints.
_CDT_TRIANGLE_BYTES = 512
# Every native output point holds 16 bytes of managed coordinates and every
# triangle a 24-byte record, so the host output copies are at most
# ceil(max(140 / 16, 512 / 24)) bytes per byte of the native window.
_CDT_HOST_BYTES_PER_NATIVE_BYTE = 22
# Fixed record, evidence and hole/segment-conversion allowance of one build.
_CDT_FIXED_BYTES = 1 << 16
# Default source-carrier fidelity bound in units of eps * coordinate scale.
# Entity support uses rational ancestry, never this distance allowance.
_CONSTRAINT_ROUNDING = 1024.0


def _remaining_request_resources(
    scope: NativeExecutionScope, /
) -> tuple[int, int, int]:
    """Read remaining capacities from the live request target, if present.

    ``NativeExecutionScope.remaining`` excludes unused capacity escrowed by an
    active ticket. Planar construction must size its next native call from that
    original ticket's remainder rather than treating the escrow as available.
    Direct owner-level use without a target retains the scope's own remainder.
    """
    target = scope._target_for_thread()
    if target is None:
        target = scope._borrowed_target_for_thread()
    if target is not None:
        return (
            target.remaining_work_units,
            target.remaining_cavity_cells,
            target.remaining_managed_bytes,
        )
    remaining = scope.remaining()
    return (
        remaining.remaining_work_units,
        remaining.remaining_cavity_cells,
        remaining.remaining_managed_bytes,
    )


@checked
def source_arrays(
    source: sources_owner.NativePlanarSource, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source vertices, source edges, and each edge's loop (-1 when embedded)."""

    region = source.region
    vertices = np.asarray(region.vertices, dtype=np.float64)
    edges = np.asarray(region.edges, dtype=np.int64)
    offsets = np.asarray(region.loop_offsets, dtype=np.int64)
    loops = np.repeat(np.arange(offsets.size - 1, dtype=np.int64), np.diff(offsets))
    if source.embedded is not None:
        embedded_vertices = np.asarray(source.embedded.vertices, dtype=np.float64)
        embedded_edges = np.asarray(source.embedded.edges, dtype=np.int64)
        edges = np.concatenate((edges, embedded_edges + vertices.shape[0]), axis=0)
        loops = np.concatenate(
            (loops, np.full((embedded_edges.shape[0],), -1, dtype=np.int64))
        )
        vertices = np.concatenate((vertices, embedded_vertices), axis=0)
    return vertices, edges, loops


def size_control_issues(
    specification: SurfaceMeshingSpec, edge_count: int, /
) -> list[str]:
    """Size controls must be uniform and scope the region or source edges."""

    controls = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    )
    if not controls or len(controls) != len(specification.size_controls):
        return ["uniform size controls only"]
    scope = specification.scope
    issues: list[str] = []
    region_scoped = False
    for control in controls:
        control_scope = control.scope
        identifiers = np.asarray(control_scope.entity_ids)
        if (control_scope.source_id, control_scope.source_revision) != (
            scope.source_id,
            scope.source_revision,
        ):
            issues.append("size-control scopes of the meshed source revision")
        elif control_scope.entity_kind is not MeshingEntityKind.GEOMETRY:
            issues.append("size controls on planar geometry entities")
        elif control_scope.entity_dimension == 2 and np.array_equal(identifiers, (0,)):
            region_scoped = True
        elif control_scope.entity_dimension != 1 or np.any(identifiers >= edge_count):
            issues.append("size controls scoped to region:0 or to source edges")
    if not region_scoped:
        issues.append("a size control on the planar region")
    return issues


@checked
def scoped_triangulation(
    points: np.ndarray,
    segments: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope,
    limits: MeshingLimits,
    holes: np.ndarray | None = None,
    min_angle: float = 0.0,
    max_area: float = math.inf,
    max_edge_length: float = math.inf,
    lattice_spacing: float = 0.0,
    max_steiner: int = 0,
    max_triangles: int | None = None,
) -> triangulation_owner.ConstrainedDelaunayTriangulation:
    """Build one CDT on a phase of the live request allowance.

    Before the native call the native target is installed and every host byte
    the build may hold is reserved: the wrapper's input copies and witness rows
    (``points + max_steiner`` rows), and the output copies, bounded by the
    output counts (at most ``min(max_triangles, 2 N)`` triangles by Euler's
    formula) or, when smaller, by the native window, since every output point
    and triangle also occupies managed native storage. Native work debits the
    request as it runs; native allocation is bounded by the remaining window and
    every insertion cavity by the remaining cavity allowance, and the measured
    peak cavity is recorded on the request ledger. A capacity refusal keeps its
    native demand evidence.
    """
    point_upper = points.shape[0] + max_steiner
    triangle_upper = 2 * point_upper
    if max_triangles is not None:
        triangle_upper = min(triangle_upper, max_triangles)
    inputs = (
        _CDT_FIXED_BYTES
        + 16 * points.shape[0]
        + 16 * segments.shape[0]
        + (0 if holes is None else 16 * holes.shape[0])
        + _CDT_WITNESS_BYTES * point_upper
    )
    outputs = _CDT_POINT_BYTES * point_upper + _CDT_TRIANGLE_BYTES * triangle_upper
    remaining_work, remaining_cavity, remaining_managed = (
        _remaining_request_resources(scope)
    )
    managed = min(limits.maximum_scratch_bytes, remaining_managed)
    cavity = min(limits.maximum_cavity_cells, remaining_cavity)
    available = max(managed - inputs, 0)
    native = available - outputs
    if outputs > _CDT_HOST_BYTES_PER_NATIVE_BYTE * native:
        native = available // (1 + _CDT_HOST_BYTES_PER_NATIVE_BYTE)
        outputs = _CDT_HOST_BYTES_PER_NATIVE_BYTE * native
    try:
        with (
            scope.batch(
                min(limits.maximum_work_units, remaining_work),
                bytes_upper=managed,
                cavity_upper=cavity,
            ) as ticket,
            ticket.native_target(),
            scope.host_workspace() as storage,
        ):
            storage.set_bound(inputs + outputs)
            triangulation = triangulation_owner.ConstrainedDelaunayTriangulation(
                points,
                segments,
                holes=holes,
                keep_convex_hull=False,
                min_angle=min_angle,
                max_area=max_area,
                max_edge_length=max_edge_length,
                lattice_spacing=lattice_spacing,
                max_steiner=max_steiner,
                max_triangles=max_triangles,
                max_cavity_cells=cavity,
                maximum_work=ticket.remaining_work_units,
                max_scratch_bytes=native,
            )
            scope.admit_cavity(int(triangulation.work_evidence[6]))
    except MeshcoreError as error:
        if error.status is not MeshcoreStatus.CAPACITY_EXCEEDED:
            raise
        raise _capacity_failure(error, limits) from error
    return triangulation


def interior_point(
    points: np.ndarray,
    loop: np.ndarray,
    limits: MeshingLimits,
    /,
    *,
    scope: NativeExecutionScope,
) -> np.ndarray:
    """A point strictly inside one simple loop: a centroid of its CDT.

    The loop CDT is built on the live request scope (``scoped_triangulation``).
    """

    local = np.arange(loop.size, dtype=np.int64)
    inside = scoped_triangulation(
        points[loop],
        np.stack((local, np.roll(local, -1)), axis=1),
        scope=scope,
        limits=limits,
    )
    return np.mean(inside.points[inside.triangles[0]], axis=0)


def _segment_distances(points: np.ndarray, segments: np.ndarray, /) -> np.ndarray:
    """Euclidean distances ``(points, segments)`` to closed segments."""

    start = segments[None, :, 0]
    direction = segments[None, :, 1] - start
    offset = points[:, None, :] - start
    squared = np.maximum(np.sum(direction * direction, axis=2), np.finfo(np.float64).tiny)
    along = np.clip(np.sum(offset * direction, axis=2) / squared, 0.0, 1.0)
    return np.linalg.norm(offset - along[..., None] * direction, axis=2)


def _field_sizes(
    points: np.ndarray,
    region_size: float,
    patch_segments: np.ndarray,
    patch_sizes: np.ndarray,
    slope: float,
    /,
) -> np.ndarray:
    """Graded target size ``min(h_region, min_e h_e + slope d_e)`` at points.

    An infinite slope (no declared growth rate) applies a patch size on its own
    edge only.
    """

    sizes = np.full((points.shape[0],), region_size, dtype=np.float64)
    if patch_sizes.size == 0 or points.shape[0] == 0:
        return sizes
    chunk = max(1, _FIELD_WORKING_ENTRIES // patch_sizes.size)
    for begin in range(0, points.shape[0], chunk):
        distance = _segment_distances(points[begin : begin + chunk], patch_segments)
        if math.isfinite(slope):
            graded = patch_sizes[None, :] + slope * distance
        else:
            graded = np.where(distance == 0.0, patch_sizes[None, :], np.inf)
        sizes[begin : begin + chunk] = np.minimum(
            sizes[begin : begin + chunk], np.min(graded, axis=1)
        )
    return sizes


def _resolve_field(
    specification: SurfaceMeshingSpec,
    vertices: np.ndarray,
    edges: np.ndarray,
    interior: np.ndarray,
    /,
) -> tuple[float, np.ndarray, np.ndarray, float, tuple[str, ...]]:
    """Region size, controlled edge rows and sizes, slope, and resolution ids.

    Region-scoped and edge-scoped controls resolve separately (their entity ids
    live in different entity sets) through the size resolution, which
    intersects hard intervals and applies priorities.
    """

    controls = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    )
    region = tuple(value for value in controls if value.scope.entity_dimension == 2)
    patch = tuple(value for value in controls if value.scope.entity_dimension == 1)
    combination = specification.size_combination
    region_field, region_report = resolve_size_controls(
        region,
        interior[None, :],
        np.zeros((1,), dtype=np.int64),
        SizeFieldDomain.EUCLIDEAN_VOLUME,
        combination=combination,
    )
    reports = [region_report.report_id]
    rows = np.zeros((0,), dtype=np.int64)
    sizes = np.zeros((0,), dtype=np.float64)
    if patch:
        rows = np.unique(
            np.concatenate(
                [np.asarray(value.scope.entity_ids, dtype=np.int64) for value in patch]
            )
        )
        midpoints = 0.5 * (vertices[edges[rows, 0]] + vertices[edges[rows, 1]])
        patch_field, patch_report = resolve_size_controls(
            patch,
            midpoints,
            rows,
            SizeFieldDomain.EUCLIDEAN_VOLUME,
            combination=combination,
        )
        sizes = np.asarray(patch_field.values, dtype=np.float64)
        reports.append(patch_report.report_id)
    rates = [
        value.maximum_growth_rate
        for value in controls
        if value.maximum_growth_rate is not None
    ]
    slope = min(rates) - 1.0 if rates else math.inf
    region_size = float(np.asarray(region_field.values)[0])
    return region_size, rows, sizes, slope, tuple(reports)


def split_constraints(
    vertices: np.ndarray,
    edges: np.ndarray,
    region_size: float,
    patch_segments: np.ndarray,
    patch_sizes: np.ndarray,
    slope: float,
    limits: MeshingLimits,
    /,
    *,
    scope: NativeExecutionScope,
    lattice_spacing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split source edges, retaining each point's binary64 rational parameter.

    ``region_size`` with the patch field is the bound every piece meets after
    dyadic subdivision. A nonzero ``lattice_spacing`` (a uniform field only)
    first cuts each edge into ``ceil(length / spacing)`` equal pieces matching
    the native interior lattice; ``region_size`` is then the construction upper,
    above the spacing, so only binary64 rounding could subdivide further.
    Every pass is charged and its host storage reserved on the live request
    ``scope`` from its current piece count before it runs.
    """

    vertex_count, edge_count = vertices.shape[0], edges.shape[0]
    patch_count = patch_sizes.size
    field_bytes = 128 * max(_FIELD_WORKING_ENTRIES, patch_count) if patch_count else 0
    # Source gathers, directions, lattice counts and per-edge pieces.
    base = 256 * (vertex_count + edge_count) + 4096
    with scope.host_workspace() as storage:
        storage.set_bound(base)
        scope.charge(work=4 * (vertex_count + edge_count))
        start = vertices[edges[:, 0]]
        direction = vertices[edges[:, 1]] - start
        if lattice_spacing:
            counts = np.minimum(
                np.ceil(np.linalg.norm(direction, axis=1) / lattice_spacing),
                limits.maximum_vertices + 1,
            ).astype(np.int64)
            point_count = vertex_count + int(np.sum(counts)) - edge_count
            if point_count > limits.maximum_vertices:
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "planar.constraints must respect limits.maximum_vertices; got excess vertices",
                    stage=MeshingStageKind.CURVE_MESHING.value,
                    requested=(("maximum_vertices", limits.maximum_vertices),),
                    achieved=(("constraint_vertices", point_count),),
                )
            pieces = int(np.sum(counts))
            # Owner, bounds and per-edge parameter ranges before concatenation.
            storage.set_bound(base + 64 * pieces)
            scope.charge(work=4 * pieces + 2 * edge_count)
            owner = np.repeat(np.arange(edge_count, dtype=np.int64), counts)
            lower = np.concatenate([np.arange(count) / count for count in counts])
            upper = np.concatenate([np.arange(1, count + 1) / count for count in counts])
        else:
            owner = np.arange(edge_count, dtype=np.int64)
            lower = np.zeros((owner.size,), dtype=np.float64)
            upper = np.ones((owner.size,), dtype=np.float64)
        while True:
            # Per piece: three corner gathers and their bank, three field
            # samples, lengths, masks and the doubled successor banks (512
            # bytes); one size-field chunk. Three samples against every patch.
            rows = owner.size
            storage.set_bound(base + 512 * rows + field_bytes)
            scope.charge(work=rows * (24 + 3 * patch_count))
            middle = 0.5 * (lower + upper)
            corners = np.concatenate(
                [
                    start[owner] + t[:, None] * direction[owner]
                    for t in (lower, middle, upper)
                ]
            )
            size = np.min(
                _field_sizes(
                    corners, region_size, patch_segments, patch_sizes, slope
                ).reshape(3, -1),
                axis=0,
            )
            long = (upper - lower) * np.linalg.norm(direction[owner], axis=1) > size
            if not np.any(long):
                break
            point_count = vertex_count + owner.size + int(np.sum(long)) - edge_count
            if point_count > limits.maximum_vertices:
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "planar.constraints must respect limits.maximum_vertices; got excess vertices",
                    stage=MeshingStageKind.CURVE_MESHING.value,
                    requested=(("maximum_vertices", limits.maximum_vertices),),
                    achieved=(("constraint_vertices", point_count),),
                )
            owner = np.concatenate((owner[~long], owner[long], owner[long]))
            lower, upper = (
                np.concatenate((lower[~long], lower[long], middle[long])),
                np.concatenate((upper[~long], middle[long], upper[long])),
            )
            del middle, corners, size, long
        # Lexsort, ordered gathers, endpoint ids and the output banks.
        rows = owner.size
        storage.set_bound(base + 512 * rows + 64 * vertex_count)
        scope.charge(work=rows * (2 * max(rows, 2).bit_length() + 16) + 4 * vertex_count)
        order = np.lexsort((lower, owner))
        owner, lower, upper = owner[order], lower[order], upper[order]
        split = lower > 0.0
        index = np.full((owner.size,), -1, dtype=np.int64)
        index[split] = vertex_count + np.arange(int(np.sum(split)), dtype=np.int64)
        first = np.where(split, index, edges[owner, 0])
        following = np.concatenate((first[1:], np.zeros((1,), dtype=np.int64)))
        last = np.where(upper == 1.0, edges[owner, 1], following)
        points = np.concatenate(
            (vertices, start[owner[split]] + lower[split, None] * direction[owner[split]])
        )
        point_sources = np.concatenate(
            (np.full(vertex_count, -1, dtype=np.int64), owner[split])
        )
        point_parameters = np.concatenate(
            (np.zeros(vertex_count, dtype=np.float64), lower[split])
        )
        return (
            points,
            np.stack((first, last), axis=1),
            owner,
            point_sources,
            point_parameters,
        )


def _loop_signs(source: sources_owner.NativePlanarSource, /) -> np.ndarray:
    """Orientation (+1 counterclockwise) of every region loop by signed area."""

    vertices = np.asarray(source.region.vertices, dtype=np.float64)
    edges = np.asarray(source.region.edges, dtype=np.int64)
    offsets = np.asarray(source.region.loop_offsets, dtype=np.int64)
    a = vertices[edges[:, 0]]
    b = vertices[edges[:, 1]]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return np.sign(np.add.reduceat(cross, offsets[:-1])).astype(np.int64)


@final
class PreparedPlanarDomain(StrictModule, NonTrainableState):
    """Graded size field and size-split constraint complex of one planar source.

    ``points`` are the source vertices followed by boundary split points;
    ``segments`` are split pieces and ``segment_sources`` their source edge IDs.
    ``point_sources`` and ``point_parameters`` retain each generated point's
    source edge and binary64 rational parameter; ``source_bounds`` limits carrier
    deviations, tightened by hard protected-feature declarations.
    ``patch_segments``/``patch_sizes`` are the size-controlled source edges and
    their resolved sizes; ``region_size`` the resolved region size and
    ``gradation_slope`` the declared growth rate minus one (``inf`` if none).
    ``construction_size`` is the uniform construction upper (``region_size``
    for graded fields). ``lattice=True`` admits, at preparation, the native
    interior lattice of an admitted uniform median: ``lattice_spacing`` is then
    nonzero and source edges are cut at that spacing; otherwise it is zero.
    ``balanced=True`` prepares the corrective construction of an admitted
    uniform median whose lattice mesh missed the median below its interval:
    source edges are cut at ``h / sqrt(2)`` and ``distribution_sites`` marks
    the interior square sites at ``0.55 h`` as charged input points.

    Preparation runs on the live request ``scope``: every host pass is charged
    and its storage reserved before it runs, and the loop interior-point CDTs
    are built by ``scoped_triangulation``.
    """

    points: np.ndarray
    segments: np.ndarray
    segment_sources: np.ndarray
    point_sources: np.ndarray
    point_parameters: np.ndarray
    source_edges: np.ndarray
    source_points: np.ndarray
    source_bounds: np.ndarray
    source_directions: np.ndarray
    source_loops: np.ndarray
    loop_signs: np.ndarray
    holes: np.ndarray
    patch_segments: np.ndarray
    patch_sizes: np.ndarray
    region_size: float = eqx.field(static=True)
    gradation_slope: float = eqx.field(static=True)
    maximum_area: float = eqx.field(static=True)
    minimum_angle_degrees: float = eqx.field(static=True)
    maximum_steiner: int = eqx.field(static=True)
    maximum_triangles: int = eqx.field(static=True)
    construction_size: float = eqx.field(static=True)
    lattice_spacing: float = eqx.field(static=True)
    distribution_sites: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: sources_owner.NativePlanarSource,
        specification: SurfaceMeshingSpec,
        /,
        *,
        scope: NativeExecutionScope,
        lattice: bool = False,
        balanced: bool = False,
    ) -> None:
        limits = specification.limits
        region = source.region
        source_rows = region.vertices.shape[0] + region.edges.shape[0]
        if source.embedded is not None:
            source_rows += (
                source.embedded.vertices.shape[0] + source.embedded.edges.shape[0]
            )
        controls = len(specification.size_controls)
        with scope.host_workspace() as storage:
            # Source banks, loop rows, orientations, directions, bounds and
            # patch gathers (512 bytes per source row); size resolution keeps
            # candidate, mask and report rows per control and sample.
            storage.set_bound(
                65536 + 512 * source_rows + 512 * controls * (source_rows + 2)
            )
            scope.charge(work=16 * source_rows)
            vertices, edges, loops = source_arrays(source)
            offsets = np.asarray(region.loop_offsets, dtype=np.int64)
            region_edges = np.asarray(region.edges, dtype=np.int64)
            interior = [
                interior_point(
                    vertices,
                    region_edges[offsets[k] : offsets[k + 1], 0],
                    limits,
                    scope=scope,
                )
                for k in range(offsets.size - 1)
            ]
            # One region sample and at most one midpoint per source edge,
            # each resolved against every control.
            scope.charge(work=16 * controls * (edges.shape[0] + 2))
            region_size, rows, sizes, slope, resolutions = _resolve_field(
                specification, vertices, edges, interior[0]
            )
            patch_segments = np.stack(
                (vertices[edges[rows, 0]], vertices[edges[rows, 1]]), 1
            )
            construction_size = (
                uniform_construction_size(specification, region_size)
                if sizes.size == 0
                else region_size
            )
            admitted = not sizes.size and uniform_median_strategy(
                specification, region_size
            )
            spacing = (
                lattice_spacing(specification, region_size)
                if lattice and not balanced and not sizes.size
                else 0.0
            )
            # The corrective construction matches its boundary spacing to a
            # square interior lattice whose diagonals fit h.
            boundary = region_size / math.sqrt(2.0) if balanced and admitted else spacing
            points, segments, sources, point_sources, point_parameters = (
                split_constraints(
                    vertices,
                    edges,
                    construction_size if boundary else region_size,
                    patch_segments,
                    sizes,
                    slope,
                    limits,
                    scope=scope,
                    lattice_spacing=boundary,
                )
            )
            holes = np.asarray(interior[1:], dtype=np.float64).reshape((-1, 2))
            fingerprinted = (
                points,
                segments,
                sources,
                point_sources,
                point_parameters,
                holes,
            )
            # Coordinate scale scan, protected-feature rows, and four visits
            # per fingerprinted element with one serialized copy of each bank.
            feature_rows = sum(
                np.asarray(feature.scope.entity_ids).size
                for feature in specification.protected_features
            )
            elements = sum(array.size for array in fingerprinted) + edges.shape[0]
            storage.set_bound(
                storage.bound
                + 2 * sum(array.nbytes for array in fingerprinted)
                + 64 * (edges.shape[0] + feature_rows)
            )
            scope.charge(work=points.size + 4 * feature_rows + 4 * elements)
            steiner = min(
                limits.maximum_work_units, limits.maximum_vertices - points.shape[0]
            )
            quality = specification.quality_target
            self.points = points
            self.segments = segments
            self.segment_sources = sources
            self.point_sources = point_sources
            self.point_parameters = point_parameters
            self.source_edges = edges
            self.source_points = vertices
            source_bound = (
                _CONSTRAINT_ROUNDING
                * np.finfo(np.float64).eps
                * max(float(np.max(np.abs(points))), 1.0)
            )
            bounds = np.full(edges.shape[0], source_bound, dtype=np.float64)
            for feature in specification.protected_features:
                if feature.hard and feature.feature_kind is not FeatureKind.CORNER:
                    rows = np.asarray(feature.scope.entity_ids, dtype=np.int64)
                    bounds[rows] = np.minimum(bounds[rows], feature.maximum_deviation)
            self.source_bounds = bounds
            self.source_directions = vertices[edges[:, 1]] - vertices[edges[:, 0]]
            self.source_loops = loops
            self.loop_signs = _loop_signs(source)
            self.holes = holes
            self.patch_segments = patch_segments
            self.patch_sizes = sizes
            self.region_size = region_size
            self.gradation_slope = slope
            self.maximum_area = (
                0.25 * math.sqrt(3.0) * construction_size * construction_size
            )
            self.minimum_angle_degrees = (
                0.0 if quality is None else math.degrees(quality.minimum_angle)
            )
            self.maximum_steiner = steiner
            self.maximum_triangles = min(limits.maximum_cells, limits.maximum_faces)
            self.construction_size = construction_size
            self.lattice_spacing = spacing
            self.distribution_sites = balanced and admitted
            self.prepared_id = canonical_fingerprint(
                {
                    "kind": "prepared-planar-domain",
                    "source": source.binding_id,
                    "specification": specification.specification_id,
                    "size_resolutions": resolutions,
                    "points": array_tree_fingerprint(self.points),
                    "segments": array_tree_fingerprint(self.segments),
                    "segment_sources": array_tree_fingerprint(self.segment_sources),
                    "point_sources": array_tree_fingerprint(self.point_sources),
                    "point_parameters": array_tree_fingerprint(self.point_parameters),
                    "source_bounds": array_tree_fingerprint(self.source_bounds),
                    "holes": array_tree_fingerprint(holes),
                    "minimum_angle_degrees": self.minimum_angle_degrees,
                    "construction_size": construction_size,
                    "lattice_spacing": spacing,
                    "distribution_sites": self.distribution_sites,
                    "maximum_area": self.maximum_area,
                    "maximum_steiner": steiner,
                    "maximum_triangles": self.maximum_triangles,
                }
            )
        # The outer planning workspace owns the prepared output after its
        # constructor's temporary source and size-field scratch is released.
        workspace = current_native_host_workspace()
        if workspace is not None:
            workspace.retain_owner(self)


    def sizes(self, points: np.ndarray, /) -> np.ndarray:
        """Target size of the graded field at points."""

        return _field_sizes(
            points,
            self.region_size,
            self.patch_segments,
            self.patch_sizes,
            self.gradation_slope,
        )

    def domain(
        self, source: sources_owner.NativePlanarSource, /
    ) -> PiecewiseLinearDomain:
        """Return the authored planar region with outward-oriented source loops."""
        return _oriented_region_domain(
            source,
            self.source_points,
            self.source_edges,
            self.source_loops,
            self.loop_signs,
        )


def _oriented_region_domain(
    source: sources_owner.NativePlanarSource,
    vertices: np.ndarray,
    edges: np.ndarray,
    loops: np.ndarray,
    signs: np.ndarray,
    /,
) -> PiecewiseLinearDomain:
    """Region facets with the outer loop counterclockwise and holes clockwise."""
    on_loop = loops >= 0
    facets = edges[on_loop]
    wanted = np.where(loops[on_loop] == 0, 1, -1)
    forward = signs[loops[on_loop]] == wanted
    facets = np.where(forward[:, None], facets, facets[:, ::-1])
    return PiecewiseLinearDomain(
        vertices,
        facets,
        np.tile(np.asarray(((0, -1),), dtype=np.int64), (facets.shape[0], 1)),
        (sources_owner.source_entity_id(source.source_revision, "region", 0),),
        source_id=source.source_id,
    )


@checked
def planar_region_domain(
    source: sources_owner.NativePlanarSource,
    /,
    *,
    scope: NativeExecutionScope,
) -> PiecewiseLinearDomain:
    """Return the authored region of ``source`` without preparing its constraints.

    The source gathers, loop orientations and oriented facet banks are charged
    and their host storage reserved on the live request ``scope``.
    """
    region = source.region
    rows = region.vertices.shape[0] + region.edges.shape[0]
    if source.embedded is not None:
        rows += source.embedded.vertices.shape[0] + source.embedded.edges.shape[0]
    with scope.host_workspace() as storage:
        storage.set_bound(256 * rows + 4096)
        scope.charge(work=12 * rows)
        vertices, edges, loops = source_arrays(source)
        return _oriented_region_domain(
            source, vertices, edges, loops, _loop_signs(source)
        )


def constrained_edges(
    triangles: np.ndarray, segment_ids: np.ndarray, sources: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Sorted vertex pairs of constrained mesh edges and their source edges."""

    carried = segment_ids >= 0
    rows, opposite = np.nonzero(carried)
    first = triangles[rows, (opposite + 1) % 3]
    second = triangles[rows, (opposite + 2) % 3]
    pairs = np.sort(np.stack((first, second), axis=1), axis=1)
    edge_sources = sources[segment_ids[rows, opposite]]
    keys, index = np.unique(pairs, axis=0, return_index=True)
    return keys, edge_sources[index]


def _association_distances(
    points: np.ndarray,
    source_points: np.ndarray,
    source_edges: np.ndarray,
    dimensions: np.ndarray,
    indices: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Return outward-certified nearest-source projection residuals."""
    original = rational_points(source_points, budget=budget)
    distances = np.empty(points.shape[0], dtype=np.float64)
    for row, point in enumerate(rational_points(points, budget=budget)):
        index = int(indices[row])
        match int(dimensions[row]):
            case 0:
                closest = original[index]
            case 1:
                first, last = source_edges[index].tolist()
                start, end = original[first], original[last]
                direction = tuple(b - a for a, b in zip(start, end, strict=True))
                squared = sum(value * value for value in direction)
                parameter = (
                    sum(
                        (p - a) * d
                        for p, a, d in zip(point, start, direction, strict=True)
                    )
                    / squared
                )
                parameter = min(max(parameter, 0), 1)
                closest = tuple(
                    a + parameter * d for a, d in zip(start, direction, strict=True)
                )
            case invalid:
                raise ValueError(
                    f"association.source_dimension must be zero or one; got {invalid}"
                )
        difference = tuple(p - q for p, q in zip(point, closest, strict=True))
        distances[row] = (
            fidelity_norm_bound(
                tuple(({(0,): value} for value in difference)), 1, budget=budget
            )
            if any(difference)
            else 0.0
        )
    return distances


@checked
def organization_and_associations(
    mesh: cell_mesh_owner.CellMesh,
    source: sources_owner.NativePlanarSource,
    prepared: PreparedPlanarDomain,
    constrained: np.ndarray,
    constrained_sources: np.ndarray,
    input_vertex_rows: np.ndarray,
    authority: ExactPlcCellGeometrySource,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[MeshPatch, ...], tuple[MeshLabel, ...], tuple[GeometryAssociation, ...]]:
    """Emit source authority from source input rows and recovered constraints.

    The CDT preserves input points before Steiner points; compaction's ``used``
    rows therefore identify authored vertices, including embedded endpoints.
    Geometry validates their support, never reconstructs their identity. Planar
    region rows retain geometric dimension two and an explicit REGION role.
    Residuals are outward bounds against the identified source primitives;
    flags require zero deviation and unique source classification.
    """
    revision = source.source_revision
    edge_set = mesh.entity_set(1)
    # ty: ignore[unresolved-attribute]
    canonical = np.asarray(mesh.connectivity.edges, dtype=np.int64)
    rows = {pair: row for row, pair in enumerate(map(tuple, np.sort(canonical, axis=1)))}
    edge_rows = np.asarray(
        [rows[tuple(pair)] for pair in constrained.tolist()], dtype=np.int64
    )
    edge_ids = np.asarray(edge_set.entity_ids, dtype=np.int64)[edge_rows]
    order = np.argsort(edge_ids, kind="stable")
    edge_ids, edge_rows = edge_ids[order], edge_rows[order]
    edge_sources = constrained_sources[order]
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    source_points, source_edges, _ = source_arrays(source)
    input_rows = np.asarray(input_vertex_rows)
    if not np.issubdtype(input_rows.dtype, np.integer):
        raise TypeError(
            "triangulation.input_rows must contain integer source rows; got a noninteger dtype"
        )
    input_rows = input_rows.astype(np.int64, copy=False)
    if (
        input_rows.shape != (points.shape[0],)
        or np.any(input_rows < 0)
        or np.unique(input_rows).size != input_rows.size
    ):
        raise ValueError(
            "triangulation.input_rows must match compacted points and source rows; got inconsistent rows"
        )
    endpoint_rows = canonical[edge_rows].reshape(-1)
    endpoint_sources = np.repeat(edge_sources, 2)
    # A vertex shared by edges of one source primitive has one residual against
    # it: evaluate each (vertex, source edge) pair once and share the bound.
    pairs, pair_index = np.unique(
        np.stack((endpoint_rows, endpoint_sources), axis=1), axis=0, return_inverse=True
    )
    pair_index = pair_index.reshape(-1)
    pair_distances = _association_distances(
        points[pairs[:, 0]],
        source_points,
        source_edges,
        np.ones((pairs.shape[0],), dtype=np.int8),
        pairs[:, 1],
        budget=budget,
    )
    edge_distances = pair_distances[pair_index].reshape((-1, 2)).max(axis=1)
    boundary = prepared.source_loops[edge_sources] >= 0
    region_distance = float(np.max(edge_distances[boundary], initial=0.0))
    vertex_dims = np.full(points.shape[0], 2, dtype=np.int8)
    vertex_sources = np.zeros(points.shape[0], dtype=np.int64)
    vertex_ambiguous = np.zeros(points.shape[0], dtype=np.bool_)
    original = input_rows < source_points.shape[0]
    vertex_dims[original], vertex_sources[original] = 0, input_rows[original]
    for row in np.unique(endpoint_rows).tolist():
        if original[row]:
            continue
        candidates = np.unique(endpoint_sources[endpoint_rows == row])
        vertex_dims[row], vertex_sources[row] = 1, candidates[0]
        vertex_ambiguous[row] = candidates.size != 1
    vertex_distances = np.full(points.shape[0], region_distance, dtype=np.float64)
    vertex_distances[original] = _association_distances(
        points[original],
        source_points,
        source_edges,
        vertex_dims[original],
        vertex_sources[original],
        budget=budget,
    )
    # Each edge vertex's residual is against an incident constrained edge's
    # source, already bounded above for that exact (vertex, source) pair.
    edge_vertices = vertex_dims == 1
    lookup = {
        pair: row
        for row, pair in enumerate(
            zip(pairs[:, 0].tolist(), pairs[:, 1].tolist(), strict=True)
        )
    }
    vertex_distances[edge_vertices] = pair_distances[
        np.fromiter(
            (
                lookup[pair]
                for pair in zip(
                    np.flatnonzero(edge_vertices).tolist(),
                    vertex_sources[edge_vertices].tolist(),
                    strict=True,
                )
            ),
            dtype=np.int64,
            count=int(np.count_nonzero(edge_vertices)),
        )
    ]
    vertex_parameters = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    witness_rows = np.asarray(authority.vertex_rows, dtype=np.int64)
    if np.any(witness_rows[edge_vertices] != vertex_sources[edge_vertices]):
        raise ValueError(
            "association.source_edges must agree with retained ancestry; got conflicting rows"
        )
    vertex_parameters[edge_vertices] = np.asarray(authority.vertex_parameters)[
        edge_vertices
    ]
    along = points[canonical[edge_rows, 1]] - points[canonical[edge_rows, 0]]
    orientations = np.sign(
        np.sum(along * prepared.source_directions[edge_sources], axis=1)
    ).astype(np.int8)
    cell_set = mesh.entity_set(2)
    cell_count = cell_set.count
    associations = (
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            revision,
            cell_set.entity_set_id,
            cell_set.entity_ids,
            tuple(
                sources_owner.source_entity_id(revision, "region", 0)
                for _ in range(cell_count)
            ),
            np.full((cell_count,), region_distance, dtype=np.float64),
            exact=region_distance == 0.0,
            source_dimensions=np.full(cell_count, 2, dtype=np.int8),
            source_indices=np.zeros(cell_count, dtype=np.int64),
            source_entity_roles=(GeometrySourceEntityRole.REGION,) * cell_count,
        ),
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            revision,
            edge_set.entity_set_id,
            edge_ids,
            tuple(
                sources_owner.source_entity_id(revision, "edge", int(edge))
                for edge in edge_sources.tolist()
            ),
            edge_distances,
            exact=bool(np.all(edge_distances == 0.0)),
            orientations=orientations,
            source_dimensions=np.full(edge_ids.size, 1, dtype=np.int8),
            source_indices=edge_sources,
            source_entity_roles=(GeometrySourceEntityRole.EDGE,) * edge_ids.size,
        ),
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            revision,
            mesh.entity_set(0).entity_set_id,
            mesh.vertex_global_ids,
            tuple(
                sources_owner.source_entity_id(
                    revision, ("vertex", "edge", "region")[int(kind)], int(index)
                )
                for kind, index in zip(vertex_dims, vertex_sources, strict=True)
            ),
            vertex_distances,
            resolved=~vertex_ambiguous,
            ambiguous=vertex_ambiguous,
            exact=not bool(np.any(vertex_ambiguous))
            and bool(np.all(vertex_distances == 0.0)),
            source_dimensions=vertex_dims,
            source_indices=vertex_sources,
            parameters=vertex_parameters,
            source_entity_roles=tuple(
                (
                    GeometrySourceEntityRole.VERTEX,
                    GeometrySourceEntityRole.EDGE,
                    GeometrySourceEntityRole.REGION,
                )[int(kind)]
                for kind in vertex_dims
            ),
        ),
    )

    def edge_scope(selected: np.ndarray, /) -> MeshingScope:
        return MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            1,
            edge_set.entity_set_id,
            selected,
        )

    loops = prepared.source_loops[edge_sources]
    patches = tuple(
        MeshPatch(f"loop:{loop}", edge_scope(edge_ids[loops == loop]))
        for loop in np.unique(loops[loops >= 0]).tolist()
    )
    labels = (
        (MeshLabel("embedded", edge_scope(edge_ids[loops < 0])),)
        if np.any(loops < 0)
        else ()
    )
    return patches, labels, associations


@checked
def graded_size_compliance(
    control: UniformSizeControl,
    policy: SizeCompliancePolicy,
    lengths: np.ndarray,
    local: np.ndarray,
    gradation: float,
    /,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]], list[str]]:
    """Requested, achieved, and failed quantities of one control's edges.

    ``local`` is the field target at each selected edge; a hard control fails
    when a target statistic, bound, the local target, or the field growth rate
    is missed beyond the compliance tolerance.
    """

    key = f"size:{control.control_id}"
    requested = [(f"{key}:target_size", control.target_size)]
    requested.extend(
        (f"{key}:{name}", value)
        for name, value in (
            ("minimum_size", control.minimum_size),
            ("maximum_size", control.maximum_size),
            ("maximum_growth_rate", control.maximum_growth_rate),
        )
        if value is not None
    )
    ratio = lengths / local
    achieved = [
        (f"{key}:edge_count", float(lengths.size)),
        (f"{key}:minimum_edge", float(np.min(lengths))),
        (f"{key}:maximum_edge", float(np.max(lengths))),
        (f"{key}:maximum_edge_to_local_size", float(np.max(ratio))),
        (f"{key}:size_field_growth_rate", gradation),
    ]
    statistics = {
        name: float(np.quantile(lengths, size_statistic_quantile(name)))
        for name in policy.target_statistics
    }
    achieved.extend((f"{key}:{name}_edge", value) for name, value in statistics.items())
    if control.strength is SizeControlStrength.SOFT:
        return requested, achieved, []

    tolerance = policy.tolerance

    issues = [
        f"target_size_{name}:{control.control_id}"
        for name, value in statistics.items()
        if abs(value - control.target_size) > tolerance(control.target_size)
    ]
    checks = (
        (
            "minimum_size",
            control.minimum_size is not None
            and float(np.min(lengths))
            < control.minimum_size - tolerance(control.minimum_size),
        ),
        (
            "maximum_size",
            control.maximum_size is not None
            and float(np.max(lengths))
            > control.maximum_size + tolerance(control.maximum_size),
        ),
        (
            "local_size",
            bool(
                np.any(
                    lengths
                    > local
                    + policy.absolute_tolerance
                    + policy.relative_tolerance * local
                )
            ),
        ),
        (
            "maximum_growth_rate",
            control.maximum_growth_rate is not None
            and gradation
            > control.maximum_growth_rate + tolerance(control.maximum_growth_rate),
        ),
    )
    issues.extend(f"{name}:{control.control_id}" for name, failed in checks if failed)
    return requested, achieved, issues


@checked
def control_edges(
    control: UniformSizeControl,
    prepared: PreparedPlanarDomain,
    edges: np.ndarray,
    local: np.ndarray,
    constrained: np.ndarray,
    constrained_sources: np.ndarray,
    /,
) -> np.ndarray:
    """Rows of the mesh edges a control governs.

    Edge-scoped controls govern the constrained edges on their source edges; a
    region control governs the edges where its size is the active target.
    """

    if control.scope.entity_dimension == 1:
        pairs = constrained[
            np.isin(constrained_sources, np.asarray(control.scope.entity_ids))
        ]
        width = np.int64(max(int(np.max(edges, initial=0)), 1) + 1)
        return np.flatnonzero(
            np.isin(edges[:, 0] * width + edges[:, 1], pairs[:, 0] * width + pairs[:, 1])
        )
    governed = np.flatnonzero(local >= prepared.region_size)
    return governed if governed.size else np.arange(edges.shape[0])


@checked
def planar_compliance(
    specification: SurfaceMeshingSpec,
    prepared: PreparedPlanarDomain,
    vertices: np.ndarray,
    triangles: np.ndarray,
    edges: np.ndarray,
    lengths: np.ndarray,
    constrained: tuple[np.ndarray, np.ndarray],
    refinement: tuple[float, int, str],
    constraint_rounding: tuple[float, float],
    /,
    *,
    scope: NativeExecutionScope,
) -> MeshingComplianceReport:
    """Independent acceptance of the published mesh against the unchanged request.

    ``edges`` are the published topology's unique edges as sorted vertex pairs
    and ``lengths`` their once-measured, already charged lengths
    (``measure_edge_lengths``); the field, gradation, every control statistic
    and bound, triangle areas and the refinement evidence are evaluated on
    them. Each remaining geometric measurement (and sizing-field distance
    query) is charged before it runs; statistics over measured lengths are
    bytes-only host packing.
    """
    minimum_angle_degrees, steiner_count, status = refinement
    edge_count, vertex_count = edges.shape[0], vertices.shape[0]
    triangle_count, patch_count = triangles.shape[0], prepared.patch_sizes.size
    if lengths.shape != (edge_count,):
        raise ValueError(
            f"planar_compliance.lengths must measure every published edge; got {lengths.shape} for {edge_count} edges"
        )
    # Semantic measurements, not array passes: one area per triangle and one
    # point-to-controlled-edge distance query per field sample (three per
    # edge, one per vertex) and controlled patch.
    work = triangle_count + (3 * edge_count + vertex_count) * patch_count
    with scope.host_workspace() as storage:
        # Endpoint, midpoint and sample banks (160 bytes per edge), targets,
        # ratios and quantile copies (64), the vertex field (8), triangle
        # corners and areas (64) and one size-field chunk per patch.
        storage.set_bound(
            224 * edge_count
            + 8 * vertex_count
            + 64 * triangle_count
            + (128 * max(_FIELD_WORKING_ENTRIES, patch_count) if patch_count else 0)
            + 4096
        )
        with scope.batch(work) as ticket:
            ticket.consume(work)
        a = vertices[edges[:, 0]]
        b = vertices[edges[:, 1]]
        local = np.min(
            prepared.sizes(np.concatenate((a, b, 0.5 * (a + b)))).reshape(3, -1), axis=0
        )
        endpoint = prepared.sizes(vertices)
        gradation = 1.0 + float(
            np.max(np.abs(endpoint[edges[:, 1]] - endpoint[edges[:, 0]]) / lengths)
        )
        requested: list[tuple[str, float]] = []
        achieved: list[tuple[str, float]] = []
        issues: list[str] = []
        for control in specification.size_controls:
            match control:
                case UniformSizeControl():
                    rows = control_edges(control, prepared, edges, local, *constrained)
                    control_requested, control_achieved, control_issues = (
                        graded_size_compliance(
                            control,
                            specification.size_compliance,
                            lengths[rows],
                            local[rows],
                            gradation,
                        )
                    )
                    requested.extend(control_requested)
                    achieved.extend(control_achieved)
                    issues.extend(control_issues)
                case _:
                    raise TypeError(
                        "specification.size_controls must contain UniformSizeControl; got an unsupported control"
                    )
        corners = vertices[triangles]
        areas = 0.5 * np.abs(
            (corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
            - (corners[:, 1, 1] - corners[:, 0, 1])
            * (corners[:, 2, 0] - corners[:, 0, 0])
        )
        maximum_area = float(np.max(areas))
    requested.extend(
        (
            ("maximum_triangle_area", prepared.maximum_area),
            ("maximum_steiner_points", prepared.maximum_steiner),
        )
    )
    achieved.extend(
        (
            ("maximum_triangle_area", maximum_area),
            ("minimum_angle", math.radians(minimum_angle_degrees)),
            ("steiner_points", steiner_count),
        )
    )
    quality = specification.quality_target
    if quality is not None:
        requested.append(("minimum_angle", quality.minimum_angle))
        if quality.hard and math.radians(minimum_angle_degrees) < quality.minimum_angle:
            issues.append(f"minimum_angle:{quality.target_id}")
    if status != "ok":
        issues.append(f"refinement_limit:{status}")
    rounding, admissible = constraint_rounding
    requested.append(("maximum_constraint_rounding", admissible))
    achieved.append(("maximum_constraint_rounding", rounding))
    if rounding > admissible:
        issues.append("constraint_rounding")
    for feature in specification.protected_features:
        requested.append(
            (
                f"protected:{feature.feature_id}:maximum_deviation",
                feature.maximum_deviation,
            )
        )
        # Source vertices are retained without rounding. Split curve vertices
        # retain the independently measured source-segment rounding bound.
        deviation = 0.0 if feature.feature_kind is FeatureKind.CORNER else rounding
        key = f"protected:{feature.feature_id}:maximum_deviation"
        achieved.append((key, deviation))
        if feature.hard and deviation > feature.maximum_deviation:
            issues.append(key)
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _distribution_seeds(
    prepared: PreparedPlanarDomain,
    specification: SurfaceMeshingSpec,
    /,
    *,
    scope: NativeExecutionScope,
) -> np.ndarray:
    """Interior sites of the corrective construction, charged as refinement points.

    Quality-only insertion resolves small features without populating the coarse
    interior. A square lattice at 0.55 h keeps its diagonals below h and avoids
    the h/2 endpoint of a 50-percent median interval. A 0.2 h constraint clearance
    avoids nearly coincident sites. Neither choice changes the declared field,
    statistics, geometry, quality target, or final certification.
    """
    if not prepared.distribution_sites:
        return np.empty((0, 2), dtype=np.float64)
    limits = specification.limits
    step = 0.55 * prepared.region_size
    # Scalar reductions precede the admission boundary; no proposal/edge arrays
    # are allocated until both work and all host working storage fit.
    lower = tuple(float(np.min(prepared.source_points[:, axis])) for axis in (0, 1))
    upper = tuple(float(np.max(prepared.source_points[:, axis])) for axis in (0, 1))
    shape = tuple(
        max(math.ceil((b - a) / step) - 1, 0) for a, b in zip(lower, upper, strict=True)
    )
    proposals = shape[0] * shape[1]
    if not proposals:
        return np.empty((0, 2), dtype=np.float64)
    edges = prepared.source_edges.shape[0]
    boundary_count = sum(int(loop) >= 0 for loop in prepared.source_loops)
    # One site construction and one visit per orientation/distance edge.
    work = proposals * (1 + boundary_count + edges)
    capacity = min(proposals, prepared.maximum_steiner)
    retained = capacity * 2 * np.dtype(np.float64).itemsize
    # Bounds cover gathered source rows, three contiguous broadcast predicate
    # operands, signs/masks, all distance intermediates and selected-point copies.
    # The output is a separately tracked native buffer, never kept+concatenated.
    fixed = 128 + 64 * (edges + boundary_count)
    per_point = 128 * (1 + edges + boundary_count)
    _, _, remaining_managed = _remaining_request_resources(scope)
    available = min(limits.maximum_scratch_bytes, remaining_managed)
    minimum_scratch = fixed + retained + per_point
    if minimum_scratch > available:
        # Preserve the owner's raw refusal before any array allocation.
        with scope.batch(0, bytes_upper=minimum_scratch):
            pass
    chunk = min(
        proposals,
        max(1, _FIELD_WORKING_ENTRIES // max(1, edges)),
        max(1, (available - fixed - retained) // per_point),
    )
    scratch = fixed + chunk * per_point
    with scope.batch(work, bytes_upper=scratch) as ticket:
        output = scope.allocate_host_array((capacity, 2), np.float64)
        boundary = prepared.source_edges[prepared.source_loops >= 0]
        a = prepared.source_points[boundary[:, 0]]
        b = prepared.source_points[boundary[:, 1]]
        segments = np.stack(
            (
                prepared.source_points[prepared.source_edges[:, 0]],
                prepared.source_points[prepared.source_edges[:, 1]],
            ),
            axis=1,
        )
        count = 0
        for begin in range(0, proposals, chunk):
            batch_size = min(chunk, proposals - begin)
            ticket.consume(batch_size * (1 + boundary_count + edges))
            index = np.arange(begin, begin + batch_size, dtype=np.int64)
            points = np.asarray(lower) + step * np.stack(
                (index // shape[1] + 1, index % shape[1] + 1), axis=1
            )
            # The predicate's native arena belongs to this admitted ticket.
            with ticket.native_target():
                signs = exact_orient2d(a[None, :, :], b[None, :, :], points[:, None, :])
            rising = (a[None, :, 1] <= points[:, None, 1]) & (
                points[:, None, 1] < b[None, :, 1]
            )
            falling = (b[None, :, 1] <= points[:, None, 1]) & (
                points[:, None, 1] < a[None, :, 1]
            )
            inside = (
                np.count_nonzero((rising & (signs > 0)) | (falling & (signs < 0)), axis=1)
                % 2
                == 1
            )
            clearance = np.min(_segment_distances(points, segments), axis=1)
            selected = inside & (clearance > 0.2 * prepared.region_size)
            end = count + int(np.count_nonzero(selected))
            if end > capacity:
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "planar.interior_sites must respect the refinement point budget; got excess sites",
                    stage=MeshingStageKind.SURFACE_MESHING.value,
                    requested=(("maximum_steiner_points", prepared.maximum_steiner),),
                    achieved=(("interior_sites", end),),
                )
            output[count:end] = points[selected]
            count = end
        return output[:count]


def _capacity_failure(error: MeshcoreError, limits: MeshingLimits, /) -> MeshingFailure:
    """Translate capacity admission without discarding the native demand ledger."""
    demand = error.batch_requested_upper
    achieved = [("native_status", int(error.status))]
    if demand is not None:
        achieved.extend((("work_units", demand[0]), ("scratch_bytes", demand[3])))
        achieved.extend(
            (f"native_batch:requested_{name}", value)
            for name, value in zip(
                ("work_units", "geometry_queries", "cavity_cells", "managed_bytes"),
                demand,
                strict=True,
            )
        )
    if error.batch_refusal_evidence is not None:
        achieved.extend(
            (f"native_batch:refusal:{index}", int(value))
            for index, value in enumerate(error.batch_refusal_evidence)
        )
    if error.memory_counters is not None:
        achieved.extend(
            (f"native_memory:{name}", value)
            for name, value in error.memory_counters.items()
        )
    if isinstance(error, NativeResourceFailure) and error.work_evidence_kind == "planar":
        achieved.extend(
            (f"native_planar:{name}", int(value))
            for name, value in zip(
                (
                    "work_units",
                    "orient_requests",
                    "incircle_requests",
                    "power_requests",
                    "insertions",
                    "flips",
                    "peak_cavity_cells",
                    "work_refusals",
                    "cavity_refusals",
                ),
                error.work_evidence,
                strict=True,
            )
        )
    return MeshingFailure(
        MeshingFailureCategory.RESOURCE_EXHAUSTED,
        "planar.refinement must respect the declared work, cavity, and scratch allowances; got excess resource demand",
        stage=MeshingStageKind.SURFACE_MESHING.value,
        requested=(
            ("maximum_work_units", limits.maximum_work_units),
            ("maximum_cavity_cells", limits.maximum_cavity_cells),
            ("maximum_scratch_bytes", limits.maximum_scratch_bytes),
        ),
        achieved=tuple(achieved),
    )


@checked
def triangulate(
    prepared: PreparedPlanarDomain,
    started: float,
    specification: SurfaceMeshingSpec,
    /,
    *,
    scope: NativeExecutionScope,
) -> tuple[triangulation_owner.ConstrainedDelaunayTriangulation, int]:
    """Retain the live request's cumulative allowance across every CDT build."""
    try:
        return _triangulate(prepared, started, specification, scope=scope)
    except MeshcoreError as error:
        if error.status is not MeshcoreStatus.CAPACITY_EXCEEDED:
            raise
        raise _capacity_failure(error, specification.limits) from error


def _triangulate(
    prepared: PreparedPlanarDomain,
    started: float,
    specification: SurfaceMeshingSpec,
    /,
    *,
    scope: NativeExecutionScope,
) -> tuple[triangulation_owner.ConstrainedDelaunayTriangulation, int]:
    """Refined CDT whose unconstrained edges respect the local target size.

    A uniform field is one native call: the admitted lattice (if any) seeds the
    carved region, or the corrective construction's charged interior sites are
    input points, and refinement bounds every unconstrained edge by the
    construction size whenever its status is ``ok``; independent compliance
    still measures every published edge, statistic and angle. A graded field
    keeps the host post-check: an unconstrained edge longer than the field at
    either end or its midpoint contributes a midpoint before the next rebuild.
    Inserted points and native refinement sites share the caller's work bound;
    every native build (``scoped_triangulation``) and host post-check runs
    under the live request scope.
    """

    limits = specification.limits
    uniform = prepared.patch_sizes.size == 0
    extra = _distribution_seeds(prepared, specification, scope=scope)
    while True:
        with scope.host_workspace() as inputs:
            # The joined input-point bank of this build.
            inputs.set_bound(16 * (prepared.points.shape[0] + extra.shape[0]) + 4096)
            joined = np.concatenate((prepared.points, extra), axis=0)
            triangulation = scoped_triangulation(
                joined,
                prepared.segments,
                scope=scope,
                limits=limits,
                holes=prepared.holes if prepared.holes.shape[0] else None,
                min_angle=prepared.minimum_angle_degrees,
                max_area=prepared.maximum_area,
                max_edge_length=prepared.construction_size if uniform else math.inf,
                lattice_spacing=prepared.lattice_spacing,
                max_steiner=max(prepared.maximum_steiner - extra.shape[0], 0),
                max_triangles=prepared.maximum_triangles,
            )
            del joined
        check_deadline(started, limits, MeshingStageKind.SURFACE_MESHING)
        if uniform:
            return triangulation, extra.shape[0]
        triangle_count = triangulation.triangles.shape[0]
        point_count = triangulation.points.shape[0]
        edge_upper = 3 * triangle_count
        depth = edge_upper.bit_length() + 2
        with scope.host_workspace() as storage:
            # Point and triangle copies (16 per point, 48 per triangle), the
            # sort-based unique and constrained edge banks (300 per triangle at
            # their transient peak), endpoint/midpoint gathers, lengths, keys,
            # masks and sampled sizes (at most 140 per edge), and one size-field
            # chunk per controlled patch.
            storage.set_bound(
                16 * point_count
                + 768 * triangle_count
                + 128 * max(_FIELD_WORKING_ENTRIES, prepared.patch_sizes.size)
                + 4096
            )
            # Point and triangle copies; unique-edge, constrained-edge and
            # free-edge membership sorts over at most 3T sides (and as many
            # constrained keys); per edge two endpoint gathers, one length, three
            # size-field samples against every controlled patch edge and their
            # minimum, and the long-edge selection.
            scope.charge(
                work=point_count
                + triangle_count
                + 4 * edge_upper * depth
                + edge_upper * (6 + 3 * prepared.patch_sizes.size)
            )
            points = np.asarray(triangulation.points, dtype=np.float64)
            triangles = np.asarray(triangulation.triangles, dtype=np.int64)
            edges = unique_edges(triangles, "triangle")
            constrained, _ = constrained_edges(
                triangles,
                np.asarray(triangulation.segment_ids, dtype=np.int64),
                prepared.segment_sources,
            )
            width = np.int64(points.shape[0])
            free = ~np.isin(
                edges[:, 0] * width + edges[:, 1],
                constrained[:, 0] * width + constrained[:, 1],
            )
            a = points[edges[:, 0]]
            b = points[edges[:, 1]]
            lengths = np.linalg.norm(b - a, axis=1)
            local = np.min(
                prepared.sizes(np.concatenate((a, b, 0.5 * (a + b)))).reshape(3, -1),
                axis=0,
            )
            long = free & (lengths > local)
            if not np.any(long):
                return triangulation, extra.shape[0]
            midpoints = 0.5 * (a[long] + b[long])
            if extra.shape[0] + midpoints.shape[0] > prepared.maximum_steiner:
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "planar.refinement must respect limits.maximum_work_units; got excess edge-length work",
                    stage=MeshingStageKind.SURFACE_MESHING.value,
                    requested=(("maximum_steiner_points", prepared.maximum_steiner),),
                    achieved=(
                        ("maximum_edge_to_local_size", float(np.max(lengths / local))),
                        ("steiner_points", extra.shape[0] + midpoints.shape[0]),
                    ),
                )
            extra = np.concatenate((extra, midpoints), axis=0)
            del points, triangles, edges, constrained, free, a, b, lengths, local
            del long, midpoints


@checked
def measure_edge_lengths(
    vertices: np.ndarray, edges: np.ndarray, /, *, scope: NativeExecutionScope
) -> np.ndarray:
    """Measure each published edge once, charged before the measurement runs.

    One semantic length per edge (the price compliance charges for an edge
    length); the median repair decision and compliance read this one bank.
    """
    count = edges.shape[0]
    with scope.host_workspace() as storage:
        # Endpoint and difference banks (48 bytes per edge) and the lengths.
        storage.set_bound(56 * count + 4096)
        with scope.batch(count) as ticket:
            ticket.consume(count)
        return np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)


@checked
def median_repair_needed(
    prepared: PreparedPlanarDomain,
    specification: SurfaceMeshingSpec,
    evidence: triangulation_owner.TriangulationEvidence,
    lengths: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope,
) -> bool:
    """Retain compliant generation; repair only an eligible undersized median.

    ``lengths`` are the candidate topology's once-measured edge lengths
    (``measure_edge_lengths``); statistics over measured lengths are
    bytes-only host packing, as in ``planar_compliance``.
    """
    if (
        prepared.patch_sizes.size
        or prepared.distribution_sites
        or not uniform_median_strategy(specification, prepared.region_size)
        or evidence.status != "ok"
    ):
        return False
    quality = specification.quality_target
    if (
        quality is not None
        and quality.hard
        and math.radians(evidence.minimum_angle_degrees) < quality.minimum_angle
    ):
        return False
    controls = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
        and control.strength is SizeControlStrength.HARD
    )
    with scope.host_workspace() as storage:
        # Targets and ratios (16 bytes per edge) and one quantile copy per
        # declared statistic and control, plus the median's own copy.
        statistics = len(specification.size_compliance.target_statistics)
        storage.set_bound(8 * lengths.size * (3 + statistics * len(controls)) + 4096)
        local = np.full(lengths.shape, prepared.region_size, dtype=np.float64)
        issues = []
        for control in controls:
            _, _, failed = graded_size_compliance(
                control, specification.size_compliance, lengths, local, 1.0
            )
            issues.extend(failed)
        return (
            bool(issues)
            and all(issue.startswith("target_size_p50:") for issue in issues)
            and float(np.quantile(lengths, 0.5)) < prepared.region_size
        )


def corrective_domain(
    source: sources_owner.NativePlanarSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedPlanarDomain,
    started: float,
    scope: NativeExecutionScope,
    /,
) -> PreparedPlanarDomain:
    """Prepare the corrective construction of an undersized lattice median.

    Feature-forced quality refinement (a thin channel, say) can outnumber the
    coarse lattice edges and pull an otherwise compliant median below its
    interval; only then is this construction prepared, charged to the same
    request by its own pre-admitted passes. Its preparation identity is emitted
    by execution's source and meshing stages; the plan keeps its own.
    """
    check_deadline(started, specification.limits, MeshingStageKind.SURFACE_MESHING)
    try:
        corrective = PreparedPlanarDomain(
            source, specification, scope=scope, balanced=True
        )
    except MeshcoreError as error:
        if error.status is not MeshcoreStatus.CAPACITY_EXCEEDED:
            raise
        raise _capacity_failure(error, specification.limits) from error
    check_deadline(started, specification.limits, MeshingStageKind.SURFACE_MESHING)
    return corrective


@checked
def constraint_coordinate_source(
    source: sources_owner.NativePlanarSource,
    prepared: PreparedPlanarDomain,
    triangulation: triangulation_owner.ConstrainedDelaunayTriangulation,
    used: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[ExactPlcCellGeometrySource, np.ndarray, np.ndarray]:
    """Compose source-edge rational ancestry and certify their carrier deviations."""
    count = triangulation.points.shape[0]
    strata = np.zeros(count, dtype=np.int64)
    rows = np.full(count, -1, dtype=np.int64)
    weights = [Fraction(0) for _ in range(count)]
    generated = np.flatnonzero(prepared.point_sources >= 0)
    strata[generated] = 1
    rows[generated] = prepared.point_sources[generated]
    for vertex in generated.tolist():
        weights[vertex] = Fraction(float(prepared.point_parameters[vertex]))

    def endpoint_parameter(vertex: int, edge: int) -> Fraction:
        if rows[vertex] == edge:
            return weights[vertex]
        first, last = prepared.source_edges[edge].tolist()
        if vertex == first:
            return Fraction(0)
        if vertex == last:
            return Fraction(1)
        raise ValueError(
            f"constraint.ancestry must bind source edge {edge}; got vertex {vertex}"
        )

    for vertex in range(triangulation.input_point_count, count):
        segment = int(triangulation.constraint_sources[vertex])
        if segment < 0:
            continue
        first, last = triangulation.constraint_parents[vertex].tolist()
        if not 0 <= first < vertex or not 0 <= last < vertex:
            raise ValueError(
                f"constraint.ancestry must be acyclic; got vertex {vertex} parents {(first, last)}"
            )
        edge = int(prepared.segment_sources[segment])
        ratio = Fraction(float(triangulation.constraint_ratios[vertex]))
        lower, upper = endpoint_parameter(first, edge), endpoint_parameter(last, edge)
        parameter = lower + ratio * (upper - lower)
        if not 0 <= ratio <= 1 or not min(lower, upper) <= parameter <= max(lower, upper):
            raise ValueError(
                f"constraint.ancestry must stay in its source interval; got vertex {vertex}"
            )
        strata[vertex], rows[vertex], weights[vertex] = 1, edge, parameter
    parameters = np.zeros((used.size, 2), dtype=np.float64)
    parameters[:, 0] = np.asarray(
        [float(weights[index]) for index in used], dtype=np.float64
    )
    ratios = tuple(
        ((weights[index].numerator, weights[index].denominator), (0, 1))
        for index in used.tolist()
    )
    authority = ExactPlcCellGeometrySource(
        prepared.source_points,
        np.empty((0, 3), dtype=np.int64),
        prepared.source_edges,
        strata[used],
        rows[used],
        parameters,
        domain_source_id=source.source_id,
        domain_source_revision=source.source_revision,
        source_triangle_ids=np.empty((0,), dtype=np.int64),
        source_triangle_bounds=np.empty((0,), dtype=np.float64),
        source_segment_ids=np.arange(prepared.source_edges.shape[0], dtype=np.int64),
        source_segment_bounds=prepared.source_bounds,
        vertex_parameter_ratios=ratios,
        budget=budget,
    )
    carriers = np.asarray(triangulation.points, dtype=np.float64)[used]
    exact_source_points = rational_points(prepared.source_points, budget=budget)
    achieved = np.zeros(prepared.source_edges.shape[0], dtype=np.float64)
    carrier_deviations = np.zeros(used.size, dtype=np.float64)
    for compact, vertex in enumerate(used.tolist()):
        edge = int(rows[vertex])
        if edge < 0:
            continue
        first, last = prepared.source_edges[edge].tolist()
        exact = tuple(
            a + weights[vertex] * (b - a)
            for a, b in zip(
                exact_source_points[first], exact_source_points[last], strict=True
            )
        )
        carrier = tuple(Fraction(float(value)) for value in carriers[compact])
        difference = tuple(a - b for a, b in zip(exact, carrier, strict=True))
        bound = fidelity_norm_bound(
            tuple(({(0,): value} for value in difference)), 1, budget=budget
        )
        carrier_deviations[compact] = bound
        achieved[edge] = max(achieved[edge], bound)
        if bound > prepared.source_bounds[edge]:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"source.edge_deviation must respect its source-fidelity bound; got edge {edge} deviation {bound}",
                stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
                entity_ids=(edge,),
                requested=(
                    (
                        f"source:edge:{edge}:maximum_deviation",
                        float(prepared.source_bounds[edge]),
                    ),
                ),
                achieved=((f"source:edge:{edge}:certified_deviation", bound),),
            )
    authority.prepare(carriers, budget=budget)
    return authority, achieved, carrier_deviations


@checked
def source_carrier_evidence(
    prepared: PreparedPlanarDomain,
    authority: ExactPlcCellGeometrySource,
    vertex_ids: np.ndarray,
    carrier_deviations: np.ndarray,
    source_deviations: np.ndarray,
    /,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    """Bind source-edge and per-carrier certified deviations to their requested bounds."""
    requested = [
        (f"source:edge:{edge}:maximum_deviation", float(bound))
        for edge, bound in enumerate(prepared.source_bounds)
    ]
    achieved = [
        (f"source:edge:{edge}:certified_deviation", float(bound))
        for edge, bound in enumerate(source_deviations)
    ]
    for vertex, edge in enumerate(np.asarray(authority.vertex_rows, dtype=np.int64)):
        if edge < 0:
            continue
        key = f"source:edge:{edge}:vertex:{vertex_ids[vertex]}"
        requested.append(
            (f"{key}:maximum_deviation", float(prepared.source_bounds[edge]))
        )
        achieved.append((f"{key}:certified_deviation", float(carrier_deviations[vertex])))
    return tuple(requested), tuple(achieved)
