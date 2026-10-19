# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Clip PLC source cells into balanced-grid pieces with corner-hex closure.

The grid is a generation variable. PLC coefficients, source strata, material
adjacency, seeds, and request bounds remain immutable. Native rational reference
clipping and octree topology retain their numerical and combinatorial evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, ExitStack
from fractions import Fraction
from itertools import product
from math import ceil, floor, gcd, lcm, log2, prod

import equinox as eqx
import jax
import numpy as np

import phydrax.discretization._cell_mesh as cell_mesh_owner
from phydrax.linalg._exact import solve_exact
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._host_banks import ancestry, entities, failure, family_host_array

from .._fingerprint import canonical_fingerprint
from .._meshcore import (
    clip_reference_tetrahedron_exact,
    ExactReferenceTetrahedronClip,
    NativeBatchTicket,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
)
from .._strict import StrictModule
from ..discretization._cell_complex import (
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._coordinate_enclosure import (
    coordinate_corner_images,
    prepared_coordinate_source_bank,
    rounded_point,
)
from ..discretization._exact_algebraic_geometry import rational_coefficient
from ..discretization._exact_plc_geometry import (
    ExactPlcCellGeometryConvexSource,
    ExactPlcCellGeometrySource,
)
from ..discretization.spatial._level_octree import refined_octree_leaves
from ..discretization.spatial._morton import morton_encode_integer_host, MortonAddressPlan
from ..geometry._domain_coverage import certify_domain_coverage, DomainCoverageCertificate
from ..geometry._global_embedding import GlobalEmbeddingCertificate
from ..typing import checked
from ._contracts import MeshingLimits
from ._dual_extraction import (
    DualExtraction,
)
from ._failures import CutStageError, CutStageRecord
from ._family_execution import (
    check_family_limits,
    coordinate_phase,
    vertex_ids,
)
from ._hex_cut_preparation import (
    compose_original_plc_supports,
    CutCornerHexConstruction,
    CutCornerHexPreparation,
    CutCornerLinkError,
    prepare_cut_corner_hexes,
    realize_cut_corner_hexes,
    require_cut_corner_quality,
)
from ._hex_generation import NativeHexGridSchedule
from ._measurements import measure_phase, NativeMeshingPhaseRecorder
from ._plc_source import PiecewiseLinearComplex
from ._volume_generation import VolumeConstruction
from .providers._native_publication import NativeCertificationRequest


class CutGridOriginPreparation(StrictModule):
    """Store rational grid-incidence exclusion parameters and performed work."""

    base: tuple[Fraction, ...] = eqx.field(static=True)
    parameter: Fraction = eqx.field(static=True)
    higher_coefficient_bound: int = eqx.field(static=True)
    source_vertex_constraints: int = eqx.field(static=True)
    source_edge_constraints: int = eqx.field(static=True)
    source_face_constraints: int = eqx.field(static=True)
    performed_work_units: int = eqx.field(static=True)

    def identity_payload(
        self,
    ) -> dict[str, int | tuple[int, int] | tuple[tuple[int, int], ...]]:
        """Return the rational exclusion parameters in deterministic JSON form."""
        return {
            "base": tuple((value.numerator, value.denominator) for value in self.base),
            "parameter": (self.parameter.numerator, self.parameter.denominator),
            "higher_coefficient_bound": self.higher_coefficient_bound,
            "source_vertex_constraints": self.source_vertex_constraints,
            "source_edge_constraints": self.source_edge_constraints,
            "source_face_constraints": self.source_face_constraints,
            "performed_work_units": self.performed_work_units,
        }


@contextmanager
@checked
def cut_stage(
    stage: str,
    origin: tuple[Fraction, ...] | None,
    receipts: list[CutStageRecord],
    *,
    scope: NativeExecutionScope,
) -> Iterator[NativeExecutionScope]:
    """Retain scoped execution on success and carry typed evidence on refusal."""
    stage_scope = NativeExecutionScope(scope.limits)
    try:
        with stage_scope:
            yield stage_scope
    except BaseException as error:
        if stage_scope.evidence is not None:
            match error:
                case CutCornerLinkError():
                    incidence = error.incidence
                case _:
                    incidence = None
            receipts.append(
                CutStageRecord(stage, origin, stage_scope.evidence, str(error), incidence)
            )
        match error:
            case Exception():
                raise CutStageError(error, tuple(receipts)) from error
            case _:
                # Signals keep their interrupt semantics; the supplied phase
                # ledger already contains their ended scope evidence.
                raise
    else:
        if stage_scope.evidence is None:
            raise ValueError(
                "cut_stage.execution must retain native scope evidence; got None"
            )
        receipts.append(CutStageRecord(stage, origin, stage_scope.evidence, None, None))


class BalancedCutHexConstruction(StrictModule):
    extraction: DualExtraction
    scaled_jacobian_lower: np.ndarray
    mean_ratio_lower: np.ndarray
    aspect_ratio_upper: np.ndarray
    octree_prefixes: np.ndarray
    octree_levels: np.ndarray
    exact_grid_origin: tuple[Fraction, ...]
    exact_grid_step: Fraction
    origin_preparation: CutGridOriginPreparation
    embedding: GlobalEmbeddingCertificate
    coverage: DomainCoverageCertificate
    stage_receipts: tuple[CutStageRecord, ...]
    cumulative_performed_work_units: int
    construction_id: str = eqx.field(static=True)


def _source_corner_bank(
    mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    *,
    budget: CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> tuple[
    tuple[tuple[Fraction, Fraction, Fraction], ...],
    tuple[tuple[int, ...], ...],
]:
    with coordinate_phase(scope=scope, budget=budget):
        elements, routes, _ = geometry.resolve(
            mesh, budget=budget, exact_source_prepared=False
        )
        coefficients = prepared_coordinate_source_bank(geometry, budget=budget)
        host_vertices, host_routes = jax.device_get(
            (
                tuple(block.vertices for block in mesh.blocks),
                routes,
            )
        )
        points: list[tuple[Fraction, Fraction, Fraction] | None] = [
            None
        ] * mesh.coordinates.shape[0]
        cells = []
        for block, element, vertices_, route in zip(
            mesh.blocks, elements, host_vertices, host_routes, strict=True
        ):
            if block.cell_kind != "tetrahedron":
                raise TypeError(
                    "source_mesh.blocks must contain tetrahedral cells; got another cell kind"
                )
            for vertices, indices in zip(
                np.asarray(vertices_), np.asarray(route), strict=True
            ):
                images = coordinate_corner_images(
                    element,
                    tuple(coefficients[int(index)] for index in indices),
                    budget=budget,
                )
                if images is None or len(images) != 4:
                    raise failure(
                        "source_geometry.corners must provide four P1 images; got unavailable images"
                    )
                for vertex, image in zip(vertices, images, strict=True):
                    point = (
                        rational_coefficient(image[0], role="source_geometry.corners"),
                        rational_coefficient(image[1], role="source_geometry.corners"),
                        rational_coefficient(image[2], role="source_geometry.corners"),
                    )
                    prior = points[int(vertex)]
                    if prior is not None and prior != point:
                        raise failure(
                            "source_geometry.vertices must have consistent incident corner images; got conflicting images"
                        )
                    points[int(vertex)] = point
                cells.append(tuple(int(vertex) for vertex in vertices))
        if any(point is None for point in points):
            raise failure(
                "source_mesh.vertices must occur in source cells; got unused coordinate coefficients"
            )
        prepared_points = tuple(point for point in points if point is not None)
    return prepared_points, tuple(cells)


def _csr(
    stencils: Sequence[tuple[tuple[int, Fraction], ...]],
    *,
    scope: NativeExecutionScope,
) -> tuple[np.ndarray, np.ndarray, tuple[Fraction, ...]]:
    count = len(stencils)
    offsets = family_host_array((count + 1,), np.int64, scope=scope)
    vertices = family_host_array(
        (sum(len(row) for row in stencils),), np.int64, scope=scope
    )
    coefficients = []
    offsets[0] = 0
    cursor = 0
    for row, stencil in enumerate(stencils):
        for vertex, coefficient in stencil:
            vertices[cursor] = vertex
            coefficients.append(coefficient)
            cursor += 1
        offsets[row + 1] = cursor
    return offsets, vertices, tuple(coefficients)


class _CutGridAdmission(StrictModule):
    source_ids: np.ndarray
    parent_ids: np.ndarray
    bounds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    pairs: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    point_bytes: int = eqx.field(static=True)
    chunk_bytes: int = eqx.field(static=True)
    scratch: int = eqx.field(static=True)


class _CutPlanes(StrictModule):
    planes: tuple[tuple[Fraction, Fraction, Fraction, Fraction], ...]
    grid_ids: tuple[tuple[int, int], ...]


class _CutVertex(StrictModule):
    point: tuple[Fraction, Fraction, Fraction]
    stencil: tuple[tuple[int, Fraction], ...]
    key: tuple[tuple[int, ...], tuple[tuple[int, int], ...]]


class _CutVertexBank(StrictModule):
    """Hold request-local streaming banks until the immutable cut result is built."""

    points: list[tuple[Fraction, Fraction, Fraction]]
    stencils: list[tuple[tuple[int, Fraction], ...]]
    keys: dict[tuple[tuple[int, ...], tuple[tuple[int, int], ...]], int]


class _CutPieceVertices(StrictModule):
    rows: tuple[int, ...]


class _CutPieces(StrictModule):
    points: tuple[tuple[Fraction, Fraction, Fraction], ...]
    stencils: tuple[tuple[tuple[int, Fraction], ...], ...]
    cells: tuple[tuple[tuple[int, ...], ...], ...]
    parents: tuple[int, ...]
    active_grid: tuple[tuple[int, ...], ...]
    work: int = eqx.field(static=True)


class _CutGridParts(StrictModule):
    mesh: cell_mesh_owner.CellMesh
    authority: ExactPlcCellGeometryConvexSource
    parents: np.ndarray
    prefixes: np.ndarray
    levels: np.ndarray
    work: int = eqx.field(static=True)


def _admit_cut_grid(
    source_mesh: cell_mesh_owner.CellMesh,
    source_geometry: CellGeometrySpec,
    source_points: tuple[tuple[Fraction, Fraction, Fraction], ...],
    source_cells: tuple[tuple[int, ...], ...],
    origin: tuple[Fraction, ...],
    step: Fraction,
    schedule: NativeHexGridSchedule,
    limits: MeshingLimits,
    *,
    scope: NativeExecutionScope,
) -> _CutGridAdmission:
    source_authority = source_geometry.exact_source
    if not isinstance(source_authority, ExactPlcCellGeometrySource):
        raise TypeError(
            "source_geometry.exact_source must be ExactPlcCellGeometrySource; got another source kind"
        )
    original = len(source_points)
    source_ids = np.asarray(jax.device_get(source_mesh.vertex_global_ids), dtype=np.int64)
    parent_ids = np.asarray(
        jax.device_get(source_mesh.entity_set(3).entity_ids), dtype=np.int64
    )
    lower_upper = []
    pairs = 0
    upper_logical = [0, 0, 0]
    for cell in source_cells:
        low = tuple(
            floor((min(source_points[v][axis] for v in cell) - origin[axis]) / step)
            for axis in range(3)
        )
        high = tuple(
            ceil((max(source_points[v][axis] for v in cell) - origin[axis]) / step) - 1
            for axis in range(3)
        )
        if min(low) < 0:
            raise ValueError(
                "grid.origin must cover source lower bounds; got a negative logical index"
            )
        pairs += prod(high[axis] - low[axis] + 1 for axis in range(3))
        for axis in range(3):
            upper_logical[axis] = max(upper_logical[axis], high[axis] + 1)
        lower_upper.append((low, high))
    depth = max(1, (max(upper_logical) - 1).bit_length())
    if depth > schedule.maximum_depth:
        raise failure(
            "grid.depth must respect schedule.maximum_depth; got an oversized logical grid",
            resource=True,
        )
    if pairs > limits.maximum_geometry_queries:
        raise failure(
            "grid.queries must respect limits.maximum_geometry_queries; got an oversized source-cell box bank",
            resource=True,
        )
    # Keep one bounded native piece beside the retained scientific banks.
    bit_bound = source_authority.maximum_bits
    point_bytes = 512 + 24 * ((bit_bound + 29) // 30)
    chunk_bytes = 16 * point_bytes + 65536
    scratch = original * 128 + chunk_bytes
    check_family_limits(limits, 0, original, 0, scratch, 32768 * pairs)
    if pairs > scope.remaining().remaining_geometry_queries:
        raise failure(
            "grid.queries must respect the remaining geometry-query allowance; got an oversized source-cell box bank",
            resource=True,
        )
    return _CutGridAdmission(
        source_ids,
        parent_ids,
        tuple(lower_upper),
        pairs,
        depth,
        point_bytes,
        chunk_bytes,
        scratch,
    )


def _prepare_cut_planes(
    images: tuple[tuple[Fraction, Fraction, Fraction], ...],
    logical: tuple[int, ...],
    origin: tuple[Fraction, ...],
    step: Fraction,
) -> _CutPlanes:
    planes = [
        (Fraction(-1), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(-1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(-1), Fraction(0)),
        (Fraction(1), Fraction(1), Fraction(1), Fraction(-1)),
    ]
    grid_ids = []
    for axis in range(3):
        column = tuple(images[index + 1][axis] - images[0][axis] for index in range(3))
        low = origin[axis] + step * logical[axis]
        high = low + step
        planes.extend(
            (
                (-column[0], -column[1], -column[2], low - images[0][axis]),
                (column[0], column[1], column[2], images[0][axis] - high),
            )
        )
        grid_ids.extend(((axis, logical[axis]), (axis, logical[axis] + 1)))
    return _CutPlanes(tuple(planes), tuple(grid_ids))


def _prepare_cut_vertex(
    support: np.ndarray,
    mask: int,
    planes: _CutPlanes,
    cell: tuple[int, ...],
    source_points: tuple[tuple[Fraction, Fraction, Fraction], ...],
    source_ids: np.ndarray,
) -> _CutVertex:
    support_indices = np.asarray(support, dtype=np.int64).reshape((-1,))
    selected = tuple(planes.planes[int(index)] for index in support_indices)
    if len(selected) != 3:
        raise failure(
            "clip.supporting_planes must name three independent planes; got another support size"
        )
    solved = solve_exact(
        [list(row[:3]) for row in selected], [[-row[3]] for row in selected]
    )
    if len(solved) != 3:
        raise ValueError(
            f"grid_parts.vertex_solve must return rank three; got {len(solved)} rows"
        )
    reference = (solved[0][0], solved[1][0], solved[2][0])
    weights = (1 - sum(reference, Fraction()), *reference)
    if min(weights) < 0:
        raise failure(
            "clip.vertex_weights must be nonnegative; got a point outside the source simplex"
        )
    stencil = tuple(
        (vertex, coefficient)
        for vertex, coefficient in zip(cell, weights, strict=True)
        if coefficient
    )
    carrier = tuple(sorted(int(source_ids[vertex]) for vertex, _ in stencil))
    grid = tuple(
        planes.grid_ids[index - 4] for index in range(4, 10) if mask & (1 << index)
    )
    point = (
        sum(
            (coefficient * source_points[vertex][0] for vertex, coefficient in stencil),
            Fraction(),
        ),
        sum(
            (coefficient * source_points[vertex][1] for vertex, coefficient in stencil),
            Fraction(),
        ),
        sum(
            (coefficient * source_points[vertex][2] for vertex, coefficient in stencil),
            Fraction(),
        ),
    )
    return _CutVertex(point, stencil, (carrier, grid))


def _retain_cut_piece_vertices(
    clipped: ExactReferenceTetrahedronClip,
    planes: _CutPlanes,
    cell: tuple[int, ...],
    source_points: tuple[tuple[Fraction, Fraction, Fraction], ...],
    admission: _CutGridAdmission,
    banks: _CutVertexBank,
    limits: MeshingLimits,
    admit_retained: Callable[[int], None],
) -> _CutPieceVertices:
    rows = []
    for position, mask in zip(
        range(clipped.supporting_planes.shape[0]), clipped.incident_planes, strict=True
    ):
        support = clipped.supporting_planes[position : position + 1].reshape(-1)
        vertex = _prepare_cut_vertex(
            support, int(mask), planes, cell, source_points, admission.source_ids
        )
        if len(vertex.stencil) == 1:
            row = vertex.stencil[0][0]
        else:
            row = banks.keys.get(vertex.key)
            if row is None:
                row = len(banks.points)
                if row + 1 > limits.maximum_vertices:
                    raise failure(
                        "clip.vertices must respect limits.maximum_vertices; got oversized retained banks",
                        resource=True,
                    )
                admit_retained(admission.point_bytes)
                banks.keys[vertex.key] = row
                banks.points.append(vertex.point)
                banks.stencils.append(vertex.stencil)
            elif banks.points[row] != vertex.point:
                raise failure(
                    "clip.vertex_restrictions must agree for shared scientific IDs; got conflicting points"
                )
        rows.append(row)
    return _CutPieceVertices(tuple(rows))


def _clip_cut_grid(
    admission: _CutGridAdmission,
    source_points: tuple[tuple[Fraction, Fraction, Fraction], ...],
    source_cells: tuple[tuple[int, ...], ...],
    origin: tuple[Fraction, ...],
    step: Fraction,
    limits: MeshingLimits,
    *,
    scope: NativeExecutionScope,
    storage: ExitStack,
) -> _CutPieces:
    allowance = scope.remaining()
    clip_ticket = scope.batch(
        32768 * admission.pairs,
        queries_upper=admission.pairs,
        cavity_upper=1,
        bytes_upper=min(limits.maximum_scratch_bytes, allowance.remaining_managed_bytes),
    )
    with clip_ticket, clip_ticket.native_target():
        workspace = storage.enter_context(scope.host_workspace())
        workspace.set_bound(admission.scratch)
        return _stream_cut_grid(
            admission,
            source_points,
            source_cells,
            origin,
            step,
            limits,
            clip_ticket,
            workspace,
        )


def _stream_cut_grid(
    admission: _CutGridAdmission,
    source_points: tuple[tuple[Fraction, Fraction, Fraction], ...],
    source_cells: tuple[tuple[int, ...], ...],
    origin: tuple[Fraction, ...],
    step: Fraction,
    limits: MeshingLimits,
    clip_ticket: NativeBatchTicket,
    workspace: NativeHostStorageWorkspace,
) -> _CutPieces:
    banks = _CutVertexBank(
        list(source_points),
        [((row, Fraction(1)),) for row in range(len(source_points))],
        {},
    )
    cut_cells: list[tuple[tuple[int, ...], ...]] = []
    cut_parents: list[int] = []
    active_grid: set[tuple[int, ...]] = set()
    work = 0
    retained_bytes = len(source_points) * 128
    face_entries = hex_count = 0

    def admit_retained(additional: int) -> None:
        nonlocal retained_bytes
        requested = retained_bytes + additional
        if requested + admission.chunk_bytes > limits.maximum_scratch_bytes:
            raise failure(
                "clip.retained_banks must fit limits.maximum_scratch_bytes beside the next bounded piece; got oversized storage",
                resource=True,
            )
        workspace.set_bound(requested + admission.chunk_bytes)
        retained_bytes = requested

    for parent, (cell, bounds) in enumerate(
        zip(source_cells, admission.bounds, strict=True)
    ):
        images = tuple(source_points[v] for v in cell)
        for logical in product(
            *(range(bounds[0][axis], bounds[1][axis] + 1) for axis in range(3))
        ):
            clip_ticket.consume(cavity_cells=1)
            planes = _prepare_cut_planes(images, logical, origin, step)
            remaining = limits.maximum_work_units - work
            if remaining <= 0:
                raise failure(
                    "clip.work must respect limits.maximum_work_units; got exhausted work",
                    resource=True,
                )
            clipped = clip_reference_tetrahedron_exact(
                planes.planes,
                maximum_scratch_bytes=limits.maximum_scratch_bytes,
                maximum_work_units=remaining,
            )
            work += clipped.work_units
            if not len(clipped.supporting_planes):
                continue
            active_grid.add(logical)
            vertex_rows = _retain_cut_piece_vertices(
                clipped,
                planes,
                cell,
                source_points,
                admission,
                banks,
                limits,
                admit_retained,
            ).rows
            loops = tuple(
                tuple(
                    vertex_rows[int(vertex)]
                    for vertex in clipped.face_vertices[
                        int(clipped.face_offsets[face]) : int(
                            clipped.face_offsets[face + 1]
                        )
                    ]
                )
                for face in range(len(clipped.face_planes))
            )
            entries = sum(len(loop) for loop in loops)
            if face_entries + entries > limits.maximum_connectivity_entries:
                raise failure(
                    "clip.face_entries must respect limits.maximum_connectivity_entries; got oversized incidence",
                    resource=True,
                )
            admit_retained(512 + 32 * entries)
            face_entries += entries
            if hex_count + len(vertex_rows) > limits.maximum_cells:
                raise failure(
                    "clip.corner_cells must respect limits.maximum_cells; got oversized closure",
                    resource=True,
                )
            hex_count += len(vertex_rows)
            cut_cells.append(loops)
            cut_parents.append(parent)
    return _CutPieces(
        tuple(banks.points),
        tuple(banks.stencils),
        tuple(cut_cells),
        tuple(cut_parents),
        tuple(sorted(active_grid)),
        work,
    )


def _assemble_cut_grid(
    source_mesh: cell_mesh_owner.CellMesh,
    source_geometry: CellGeometrySpec,
    original: int,
    admission: _CutGridAdmission,
    pieces: _CutPieces,
    limits: MeshingLimits,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> _CutGridParts:
    if not pieces.cells:
        raise failure(
            "clip.pieces must contain positive-volume cells; got an empty partition"
        )
    coordinates = family_host_array((len(pieces.points), 3), np.float64, scope=scope)
    for row, point in enumerate(pieces.points):
        coordinates[row] = rounded_point(point)
    piece_ids = family_host_array((len(pieces.cells),), np.int64, scope=scope)
    piece_ids[:] = np.arange(len(pieces.cells), dtype=np.int64)
    poly = cell_mesh_owner.CellMesh.from_polyhedra(
        coordinates,
        pieces.cells,
        vertex_global_ids=vertex_ids(source_mesh, len(pieces.points) - original),
        cell_global_ids=piece_ids,
        numeric_version=source_mesh.numeric_version,
    )
    connectivity = poly.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError(
            "grid_parts.connectivity must be polyhedral; got another connectivity"
        )
    if (
        connectivity.edge_count > limits.maximum_edges
        or connectivity.face_count > limits.maximum_faces
    ):
        raise failure(
            "clip.topology must respect limits.maximum_edges and limits.maximum_faces; got oversized incidence",
            resource=True,
        )
    offsets, vertices, coefficients = _csr(pieces.stencils, scope=scope)
    parents = family_host_array((len(pieces.parents),), np.int64, scope=scope)
    # The carrier groups cells by arity. Scientific piece IDs preserve the
    # streamed parent association across that row permutation.
    ordered_piece_ids = np.asarray(
        jax.device_get(poly.entity_set(3).entity_ids), dtype=np.int64
    )
    for row, piece_id in enumerate(ordered_piece_ids):
        parents[row] = pieces.parents[int(piece_id)]
    with coordinate_phase(scope=scope, budget=budget):
        authority = ExactPlcCellGeometryConvexSource(
            source_mesh,
            source_geometry,
            offsets,
            vertices,
            coefficients,
            poly,
            admission.parent_ids[parents],
            budget=budget,
        )
    integer = family_host_array((len(pieces.active_grid), 3), np.int64, scope=scope)
    integer[:] = pieces.active_grid
    depth = admission.depth
    address = MortonAddressPlan(
        (0.0, 0.0, 0.0), tuple(float(1 << depth) for _ in range(3)), depth
    )
    codes = morton_encode_integer_host(integer, depth)
    refined = [
        np.unique(codes >> np.uint64(3 * (depth - level))) for level in range(depth)
    ]
    prefixes, levels, _ = refined_octree_leaves(address, refined, balanced=True)
    return _CutGridParts(poly, authority, parents, prefixes, levels, pieces.work)


def _grid_parts(
    source_mesh: cell_mesh_owner.CellMesh,
    source_geometry: CellGeometrySpec,
    source_points: tuple[tuple[Fraction, Fraction, Fraction], ...],
    source_cells: tuple[tuple[int, ...], ...],
    origin: tuple[Fraction, ...],
    step: Fraction,
    schedule: NativeHexGridSchedule,
    limits: MeshingLimits,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    workspace: NativeHostStorageWorkspace,
) -> _CutGridParts:
    admission = _admit_cut_grid(
        source_mesh,
        source_geometry,
        source_points,
        source_cells,
        origin,
        step,
        schedule,
        limits,
        scope=scope,
    )
    starting = workspace.bound
    workspace.set_bound(starting + admission.scratch)
    try:
        # The token is born in the ticket's pool and remains live through
        # assembly. A parent-pool token cannot grow under a full reservation.
        workspace.set_bound(starting)
        with ExitStack() as storage:
            pieces = _clip_cut_grid(
                admission,
                source_points,
                source_cells,
                origin,
                step,
                limits,
                scope=scope,
                storage=storage,
            )
            # Assembly packing belongs to the root pool; the ticket-local
            # workspace continues to admit the live immutable input banks.
            with scope.host_workspace():
                result = _assemble_cut_grid(
                    source_mesh,
                    source_geometry,
                    len(source_points),
                    admission,
                    pieces,
                    limits,
                    scope=scope,
                    budget=budget,
                )
                workspace.retain_owner(result)
    except BaseException:
        workspace.set_bound(starting)
        raise
    return result


def _canonical_cut_origin(
    mesh: cell_mesh_owner.CellMesh,
    points: tuple[tuple[Fraction, Fraction, Fraction], ...],
    base: tuple[Fraction, ...],
    step: Fraction,
    limits: MeshingLimits,
    *,
    scope: NativeExecutionScope,
) -> tuple[tuple[Fraction, ...], CutGridOriginPreparation]:
    """Exclude grid/source coincidences by an exact integer-polynomial bound.

    Origin=base+step*(t,t²,t³). For every nonzero incidence polynomial,
    clearing its rational denominators gives integer coefficients.
    If 0<t<1/(1+sum(abs(higher coefficients))), its first nonzero term
    dominates all later terms. This holds for every integer grid address;
    no trial clipping, source movement, or field-iteration allowance is used.
    """
    maximum = 0
    work = 0
    constraints = (
        3 * len(points) + 3 * mesh.entity_set(1).count + mesh.entity_set(2).count
    )
    check_family_limits(limits, 0, 0, 0, 32768, 256 * constraints)
    origin_ticket = scope.batch(256 * constraints)

    with origin_ticket:

        def admit(coefficients: Sequence[Fraction]) -> None:
            nonlocal maximum, work
            origin_ticket.consume(32)
            work += 32
            denominator = lcm(*(value.denominator for value in coefficients))
            integers = tuple(
                value.numerator * (denominator // value.denominator)
                for value in coefficients
            )
            divisor = gcd(*integers)
            if divisor:
                integers = tuple(value // divisor for value in integers)
            maximum = max(maximum, sum(abs(value) for value in integers[1:]))

        for point in points:
            for axis in range(3):
                higher = [Fraction(0)] * 3
                higher[axis] = -step
                admit((point[axis] - base[axis], *higher))
        connectivity = mesh.connectivity
        if not isinstance(connectivity, TetrahedralConnectivity):
            raise TypeError(
                "grid_origin.connectivity must be tetrahedral; got another connectivity"
            )
        edge_count = 0
        for first, second in np.asarray(connectivity.edges).tolist():
            a, b = points[first], points[second]
            direction = tuple(y - x for x, y in zip(a, b, strict=True))
            for x, y in ((0, 1), (0, 2), (1, 2)):
                if direction[x] == direction[y] == 0:
                    continue
                higher = [Fraction(0)] * 3
                higher[x], higher[y] = step * direction[y], -step * direction[x]
                constant = direction[y] * (base[x] - a[x]) - direction[x] * (
                    base[y] - a[y]
                )
                admit((constant, *higher))
                edge_count += 1
        faces = entities(mesh, 2)
        for row in faces.tolist():
            a, b, c = (points[vertex] for vertex in row)
            u, v = (
                tuple(y - x for x, y in zip(a, b, strict=True)),
                tuple(y - x for x, y in zip(a, c, strict=True)),
            )
            normal = (
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            )
            if not any(normal):
                raise failure(
                    "grid_origin.source_faces must have nonzero rational normals; got a degenerate face"
                )
            constant = sum(
                (normal[axis] * (base[axis] - a[axis]) for axis in range(3)), Fraction(0)
            )
            admit((constant, *(step * value for value in normal)))
    parameter = Fraction(1, maximum + 2)
    origin = tuple(base[axis] + step * parameter ** (axis + 1) for axis in range(3))
    proof = CutGridOriginPreparation(
        base, parameter, maximum, 3 * len(points), edge_count, len(faces), work
    )
    return origin, proof


class _CutAncestry(StrictModule):
    supports: np.ndarray
    parents: np.ndarray
    dimensions: tuple[np.ndarray, ...]
    rows: tuple[np.ndarray, ...]
    boundary: np.ndarray
    valences: np.ndarray
    singular: np.ndarray


def _prepare_cut_ancestry(
    source_mesh: cell_mesh_owner.CellMesh,
    actual: CutCornerHexConstruction,
    prepared: CutCornerHexPreparation,
    parents: np.ndarray,
    *,
    scope: NativeExecutionScope,
) -> _CutAncestry:
    original_offsets, original_vertices, _ = compose_original_plc_supports(
        prepared, scope=scope
    )
    supports = family_host_array(
        (actual.mesh.coordinates.shape[0], 4), np.int64, scope=scope
    )
    supports.fill(-1)
    for row in range(supports.shape[0]):
        indices = original_vertices[original_offsets[row] : original_offsets[row + 1]]
        supports[row, : len(indices)] = indices
    parent_cells = parents[prepared.parent_cells]
    dimensions, rows = ancestry(source_mesh, actual.mesh, supports)
    source_connectivity = source_mesh.connectivity
    if not isinstance(source_connectivity, TetrahedralConnectivity):
        raise TypeError(
            "balanced_cut.source_connectivity must remain tetrahedral; got another connectivity"
        )
    faces = entities(source_mesh, 2)
    incidence = np.bincount(
        np.asarray(jax.device_get(source_connectivity.cell_faces)).reshape(-1),
        minlength=len(faces),
    )
    target_faces = entities(actual.mesh, 2)
    on_boundary = (dimensions[2] == 2) & (
        incidence[np.minimum(rows[2], len(faces) - 1)] == 1
    )
    boundary = family_host_array(
        (actual.mesh.coordinates.shape[0],), np.bool_, scope=scope
    )
    boundary.fill(False)
    boundary[target_faces[on_boundary].reshape(-1)] = True
    valences = np.bincount(
        prepared.hexes.reshape(-1), minlength=actual.mesh.coordinates.shape[0]
    ).astype(np.int64)
    singular = np.flatnonzero((~boundary) & (valences != 8)).astype(np.int64)
    return _CutAncestry(
        supports, parent_cells, dimensions, rows, boundary, valences, singular
    )


def _assemble_balanced_cut(
    source_mesh: cell_mesh_owner.CellMesh,
    source_geometry: CellGeometrySpec,
    schedule: NativeHexGridSchedule,
    origin: tuple[Fraction, ...],
    step: Fraction,
    origin_preparation: CutGridOriginPreparation,
    actual: CutCornerHexConstruction,
    grid: _CutGridParts,
    source_ancestry: _CutAncestry,
    coverage: DomainCoverageCertificate,
    receipts: tuple[CutStageRecord, ...],
) -> BalancedCutHexConstruction:
    identity = canonical_fingerprint(
        {
            "kind": "original-plc-balanced-cut-hex",
            "source_geometry": cell_geometry_id(source_geometry),
            "source_topology": source_mesh.topology_id,
            "schedule": schedule.schedule_id,
            "origin": tuple((value.numerator, value.denominator) for value in origin),
            "step": (step.numerator, step.denominator),
            "mesh": actual.mesh.mesh_id,
            "geometry": cell_geometry_id(actual.geometry),
            "embedding": actual.embedding.certificate_id,
            "coverage": coverage.certificate_id,
            "origin_preparation": origin_preparation.identity_payload(),
        }
    )
    performed = sum(
        int(receipt.execution.work_counters.construction_work) for receipt in receipts
    )
    extraction = DualExtraction(
        actual.mesh,
        source_mesh,
        source_ancestry.supports,
        source_ancestry.parents,
        source_ancestry.dimensions,
        source_ancestry.rows,
        actual.validity,
        source_ancestry.valences,
        source_ancestry.boundary,
        source_ancestry.singular,
        None,
        performed,
        actual.geometry,
        source_geometry,
    )
    return BalancedCutHexConstruction(
        extraction,
        actual.scaled_jacobian_lower,
        actual.mean_ratio_lower,
        actual.aspect_ratio_upper,
        grid.prefixes,
        grid.levels,
        origin,
        step,
        origin_preparation,
        actual.embedding,
        coverage,
        receipts,
        performed,
        identity,
    )


@checked
def generate_balanced_cut_hexes(
    complex_: PiecewiseLinearComplex,
    construction: VolumeConstruction,
    schedule: NativeHexGridSchedule,
    limits: MeshingLimits,
    target_size: float,
    /,
    *,
    source_geometry: CellGeometrySpec,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> BalancedCutHexConstruction:
    """Construct cut-grid hexes and certify their maps against the PLC source."""
    if schedule.route != "balanced_grid":
        raise ValueError("schedule.route must be balanced_grid; got another hex route")
    if not isinstance(source_geometry.exact_source, ExactPlcCellGeometrySource):
        raise TypeError(
            "source_geometry.exact_source must retain a PLC construction; got another coordinate source"
        )
    if complex_.boundary == "fixed":
        lengths = np.diff(np.asarray(complex_.polygon_offsets))
        incompatible = np.flatnonzero(lengths != 4)
        if incompatible.size:
            raise failure(
                f"complex.boundary must contain quadrilateral polygons for unchanged pure-hex closure; got incompatible rows {tuple(int(row) for row in incompatible)}"
            )
        raise failure(
            "complex.boundary must permit subdivision for the selected corner template; got fixed quadrilateral faces"
        )
    if not np.isfinite(target_size) or target_size <= 0:
        raise ValueError(
            f"cut_grid.target_size must be finite and positive; got {target_size!r}"
        )
    with scope.host_workspace() as workspace:
        receipts = []
        with cut_stage(
            "source_and_origin_preparation", None, receipts, scope=scope
        ) as stage_scope:
            source_mesh = construction.mesh
            points, cells = _source_corner_bank(
                source_mesh, source_geometry, budget=budget, scope=stage_scope
            )
            exponent = floor(log2(target_size)) + 1
            step = Fraction(2) ** exponent
            base = tuple(
                Fraction(floor(min(point[axis] for point in points) / step) - 1) * step
                for axis in range(3)
            )
            origin, origin_preparation = _canonical_cut_origin(
                source_mesh, points, base, step, limits, scope=stage_scope
            )
        request = NativeCertificationRequest(
            MeshCertificationSchedule("volume_plc"),
            source_geometry.exact_source.domain_source_id,
            source_geometry.exact_source.domain_source_revision,
            limits,
            domain=construction.domain,
        )
        with (
            cut_stage("source_clipping", origin, receipts, scope=scope) as stage_scope,
            measure_phase(record_phase, "construction"),
        ):
            cut_grid = _grid_parts(
                source_mesh,
                source_geometry,
                points,
                cells,
                origin,
                step,
                schedule,
                limits,
                scope=stage_scope,
                budget=budget,
                workspace=workspace,
            )
            poly, authority, parents = cut_grid.mesh, cut_grid.authority, cut_grid.parents
        with (
            cut_stage("corner_closure", origin, receipts, scope=scope) as stage_scope,
            measure_phase(record_phase, "topology_construction"),
        ):
            prepared = prepare_cut_corner_hexes(
                poly,
                authority,
                limits,
                scope=stage_scope,
                budget=budget,
                workspace=workspace,
            )
        with (
            cut_stage(
                "exact_maps_embedding_quality", origin, receipts, scope=scope
            ) as stage_scope,
            measure_phase(record_phase, "curving"),
        ):
            actual = realize_cut_corner_hexes(
                prepared,
                limits,
                certificate_limits=request.certificate_limits,
                scope=stage_scope,
                budget=budget,
            )
            require_cut_corner_quality(actual, schedule)
        with cut_stage(
            "original_scientific_ancestry", origin, receipts, scope=scope
        ) as stage_scope:
            source_ancestry = _prepare_cut_ancestry(
                source_mesh, actual, prepared, parents, scope=stage_scope
            )

        with (
            cut_stage(
                "original_domain_coverage", origin, receipts, scope=scope
            ) as stage_scope,
            measure_phase(record_phase, "certification"),
        ):
            with coordinate_phase(scope=stage_scope, budget=budget):
                coverage = certify_domain_coverage(
                    actual.mesh,
                    actual.geometry,
                    construction.domain,
                    construction.cell_regions[source_ancestry.parents],
                    embedding=actual.embedding,
                    limits=request.certificate_limits,
                    budget=budget,
                )
            if coverage.status != "certified":
                raise failure(
                    f"cut_grid.coverage.status must be certified; got findings {tuple(finding.check for finding in coverage.findings)}"
                )
        return _assemble_balanced_cut(
            source_mesh,
            source_geometry,
            schedule,
            origin,
            step,
            origin_preparation,
            actual,
            cut_grid,
            source_ancestry,
            coverage,
            tuple(receipts),
        )
