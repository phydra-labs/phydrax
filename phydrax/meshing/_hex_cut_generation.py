# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Balanced-grid source-cell clipping with scientific corner-hex closure.

The grid is a generation variable. Original PLC coordinates, source strata,
material adjacency, seeds and request bounds remain immutable. Exact native
reference clipping and canonical octree topology own the numerical/combinatorial
kernels. An unresolved source-cell link is never replaced by a dual extraction.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import ceil, floor, gcd, lcm, log2, prod

import numpy as np

from .._fingerprint import canonical_fingerprint
from .._meshcore import (
    clip_reference_tetrahedron_exact,
    current_native_execution_budget,
    current_native_host_workspace,
    NativeExecutionBudget,
    NativeExecutionEvidence,
)
from ..discretization import CellMesh
from ..discretization._cell_complex import (
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_enclosure import (
    _solve_exact,
    coordinate_corner_images,
    prepared_coordinate_source_bank,
    rounded_point,
)
from ..discretization._exact_plc_geometry import (
    ExactPlcCellGeometryConvexSource,
    ExactPlcCellGeometrySource,
)
from ..discretization.spatial._level_octree import refined_octree_leaves
from ..discretization.spatial._morton import morton_encode_integer_host, MortonAddressPlan
from ..geometry._mesh_certificates import (
    DomainCoverageCertificate,
    GlobalEmbeddingCertificate,
)
from ._certification import MeshCertificationSchedule
from ._contracts import MeshingLimits
from ._hex_cut_preparation import (
    compose_original_plc_supports,
    CutCornerLinkError,
    CutCornerLinkWitness,
    prepare_cut_corner_hexes,
    realize_cut_corner_hexes,
    require_cut_corner_quality,
)
from ._hex_generation import NativeHexGridSchedule
from ._measurements import measure_phase, NativeMeshingPhaseRecorder
from ._quad_generation import (
    _ancestry,
    _budget,
    _entities,
    _failure,
    _family_host_array,
    _vertex_ids,
    DualExtraction,
)
from ._volume_generation import PiecewiseLinearComplex, VolumeConstruction
from .providers._native_publication import NativeCertificationRequest


@dataclass(frozen=True, slots=True)
class CutGridOriginProof:
    base: tuple[Fraction, ...]
    parameter: Fraction
    higher_coefficient_bound: int
    source_vertex_constraints: int
    source_edge_constraints: int
    source_face_constraints: int
    performed_work_units: int


@dataclass(frozen=True, slots=True)
class CutStageReceipt:
    stage: str
    origin: tuple[Fraction, ...] | None
    execution: NativeExecutionEvidence
    failure: str | None
    link_witness: CutCornerLinkWitness | None


@contextmanager
def _cut_stage(
    stage: str,
    origin: tuple[Fraction, ...] | None,
    limits: MeshingLimits,
    receipts: list[CutStageReceipt],
) -> Iterator[None]:
    """Retain the actual scoped execution on success and every failure prefix."""
    budget = NativeExecutionBudget(
        max_work=limits.maximum_work_units,
        max_geometry_queries=limits.maximum_geometry_queries,
        max_cavity_cells=limits.maximum_cavity_cells,
        max_scratch_bytes=limits.maximum_scratch_bytes,
        max_wall_seconds=limits.maximum_wall_seconds,
    )
    try:
        with budget:
            yield
    except BaseException as error:
        if budget.evidence is not None:
            witness = error.witness if isinstance(error, CutCornerLinkError) else None
            receipts.append(
                CutStageReceipt(stage, origin, budget.evidence, str(error), witness)
            )
            setattr(error, "cut_failure_prefix", tuple(receipts))
        raise
    else:
        if budget.evidence is None:
            raise RuntimeError("The actual cut stage lost its native execution receipt.")
        receipts.append(CutStageReceipt(stage, origin, budget.evidence, None, None))


@dataclass(frozen=True, slots=True)
class BalancedCutHexConstruction:
    extraction: DualExtraction
    scaled_jacobian_lower: np.ndarray
    mean_ratio_lower: np.ndarray
    aspect_ratio_upper: np.ndarray
    octree_prefixes: np.ndarray
    octree_levels: np.ndarray
    exact_grid_origin: tuple[Fraction, ...]
    exact_grid_step: Fraction
    origin_proof: CutGridOriginProof
    embedding: GlobalEmbeddingCertificate
    coverage: DomainCoverageCertificate
    stage_receipts: tuple[CutStageReceipt, ...]
    cumulative_performed_work_units: int
    construction_id: str


def _source_corner_bank(
    mesh: CellMesh, geometry: CellGeometrySpec
) -> tuple[
    tuple[tuple[Fraction, Fraction, Fraction], ...],
    tuple[tuple[int, ...], ...],
]:
    elements, routes, _ = geometry.resolve(mesh)
    coefficients = prepared_coordinate_source_bank(geometry)
    points = [None] * mesh.coordinates.shape[0]
    cells = []
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if block.cell_kind != "tetrahedron":
            raise TypeError(
                "Balanced PLC cuts consume the genuine original tetrahedral source carrier."
            )
        for vertices, indices in zip(
            np.asarray(block.vertices), np.asarray(route), strict=True
        ):
            images = coordinate_corner_images(
                element, tuple(coefficients[int(index)] for index in indices)
            )
            if images is None or len(images) != 4:
                raise _failure(
                    "Original PLC source has no complete exact P1 corner-image law."
                )
            for vertex, image in zip(vertices, images, strict=True):
                prior = points[int(vertex)]
                if prior is not None and prior != image:
                    raise _failure(
                        "Original source SCI vertex has conflicting complete exact source images."
                    )
                points[int(vertex)] = image
            cells.append(tuple(int(vertex) for vertex in vertices))
    if any(point is None for point in points):
        raise _failure("Original source has an unused coordinate coefficient vertex.")
    prepared_points = tuple(point for point in points if point is not None)
    if len(prepared_points) != len(points):
        raise RuntimeError("Original source point narrowing lost a vertex.")
    return prepared_points, tuple(cells)


def _csr(
    stencils: Sequence[tuple[tuple[int, Fraction], ...]],
) -> tuple[np.ndarray, np.ndarray, tuple[Fraction, ...]]:
    count = len(stencils)
    offsets = _family_host_array((count + 1,), np.int64)
    vertices = _family_host_array((sum(len(row) for row in stencils),), np.int64)
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


def _grid_parts(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    source_points: tuple[tuple[Fraction, Fraction, Fraction], ...],
    source_cells: tuple[tuple[int, ...], ...],
    origin: tuple[Fraction, ...],
    step: Fraction,
    schedule: NativeHexGridSchedule,
    limits: MeshingLimits,
) -> tuple[
    CellMesh,
    ExactPlcCellGeometryConvexSource,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    source_authority = source_geometry.exact_source
    if not isinstance(source_authority, ExactPlcCellGeometrySource):
        raise TypeError("Balanced source cuts require retained exact PLC authority.")
    original = len(source_points)
    source_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
    parent_ids = np.asarray(source_mesh.entity_set(3).entity_ids, dtype=np.int64)
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
                "Exact generation grid does not cover its original source lower bounds."
            )
        pairs += prod(high[axis] - low[axis] + 1 for axis in range(3))
        for axis in range(3):
            upper_logical[axis] = max(upper_logical[axis], high[axis] + 1)
        lower_upper.append((low, high))
    depth = max(1, (max(upper_logical) - 1).bit_length())
    if depth > schedule.maximum_depth:
        raise _failure(
            "Original source-cell cut grid exceeds its selected maximum depth.",
            resource=True,
        )
    if pairs > limits.maximum_geometry_queries:
        raise _failure(
            "Original source-cell box queries exceed the original query bound.",
            resource=True,
        )
    # Stream one native bounded piece while retaining actual SCI banks. The
    # request's bit cap still bounds every retained rational; empty or welded
    # pieces do not reserve sixteen fictitious new vertices apiece.
    bit_bound = source_authority.maximum_bits
    point_bytes = 512 + 24 * ((bit_bound + 29) // 30)
    chunk_bytes = 16 * point_bytes + 65536
    scratch = original * 128 + chunk_bytes
    _budget(limits, 0, original, 0, scratch, 32768 * pairs)
    execution = current_native_execution_budget()
    workspace = current_native_host_workspace()
    if execution is not None:
        if workspace is None:
            raise _failure(
                "Balanced source cuts require the original host-storage owner.",
                resource=True,
            )
        allowance = execution.remaining()
        if pairs > allowance.remaining_geometry_queries:
            raise _failure(
                "Original source-cell queries exceed the remaining source allowance.",
                resource=True,
            )
        execution.admit_work_bound(32768 * pairs)
    starting = 0 if workspace is None else workspace.bound
    if workspace is not None:
        workspace.set_bound(starting + scratch)
    points = list(source_points)
    stencils: list[tuple[tuple[int, Fraction], ...]] = [
        ((row, Fraction(1)),) for row in range(original)
    ]
    keys = {}
    cut_cells, cut_parents, active_grid = [], [], set()
    work = 0
    retained_bytes = original * 128
    face_entries = 0

    def admit_retained(additional: int) -> None:
        nonlocal retained_bytes
        requested = retained_bytes + additional
        if requested + chunk_bytes > limits.maximum_scratch_bytes:
            raise _failure(
                "Actual retained scientific cut banks plus next bounded native piece exceed original scratch.",
                resource=True,
            )
        if workspace is not None:
            workspace.set_bound(starting + requested + chunk_bytes)
        retained_bytes = requested

    hex_count = 0
    try:
        for parent, (cell, bounds) in enumerate(
            zip(source_cells, lower_upper, strict=True)
        ):
            images = tuple(source_points[v] for v in cell)
            for logical in product(
                *(range(bounds[0][axis], bounds[1][axis] + 1) for axis in range(3))
            ):
                if execution is not None:
                    execution.admit_cavity(1)
                planes: list[tuple[Fraction, Fraction, Fraction, Fraction]] = [
                    (Fraction(-1), Fraction(0), Fraction(0), Fraction(0)),
                    (Fraction(0), Fraction(-1), Fraction(0), Fraction(0)),
                    (Fraction(0), Fraction(0), Fraction(-1), Fraction(0)),
                    (Fraction(1), Fraction(1), Fraction(1), Fraction(-1)),
                ]
                grid_plane_ids = []
                for axis in range(3):
                    column = tuple(
                        images[index + 1][axis] - images[0][axis] for index in range(3)
                    )
                    low = origin[axis] + step * logical[axis]
                    high = low + step
                    planes.extend(
                        (
                            (
                                -column[0],
                                -column[1],
                                -column[2],
                                low - images[0][axis],
                            ),
                            (
                                column[0],
                                column[1],
                                column[2],
                                images[0][axis] - high,
                            ),
                        )
                    )
                    grid_plane_ids.extend(
                        ((axis, logical[axis]), (axis, logical[axis] + 1))
                    )
                remaining = limits.maximum_work_units - work
                if remaining <= 0:
                    raise _failure(
                        "Exact source-cell clipping exhausts the original work bound.",
                        resource=True,
                    )
                clipped = clip_reference_tetrahedron_exact(
                    tuple(planes),
                    maximum_scratch_bytes=limits.maximum_scratch_bytes,
                    maximum_work_units=remaining,
                )
                work += clipped.work_units
                if not len(clipped.supporting_planes):
                    continue
                active_grid.add(logical)
                vertex_rows = []
                for support, mask_value in zip(
                    clipped.supporting_planes,
                    clipped.incident_planes,
                    strict=True,
                ):
                    support_indices = np.asarray(support, dtype=np.int64).reshape((-1,))
                    selected = tuple(planes[int(index)] for index in support_indices)
                    if len(selected) != 3:
                        raise _failure(
                            "Native cut vertex lacks three exact supporting planes."
                        )
                    solved = _solve_exact(
                        [list(row[:3]) for row in selected],
                        [[-row[3]] for row in selected],
                    )
                    if len(solved) != 3:
                        raise RuntimeError(
                            "Exact cut vertex solve returned the wrong rank."
                        )
                    reference = (solved[0][0], solved[1][0], solved[2][0])
                    weights = (
                        1 - sum(reference, Fraction()),
                        *reference,
                    )
                    if min(weights) < 0:
                        raise _failure(
                            "Native original source-plane witness leaves its source simplex."
                        )
                    stencil: tuple[tuple[int, Fraction], ...] = tuple(
                        (vertex, coefficient)
                        for vertex, coefficient in zip(cell, weights, strict=True)
                        if coefficient
                    )
                    mask = int(mask_value)
                    carrier = tuple(
                        sorted(int(source_ids[vertex]) for vertex, _ in stencil)
                    )
                    grid = tuple(
                        grid_plane_ids[index - 4]
                        for index in range(4, 10)
                        if mask & (1 << index)
                    )
                    point: tuple[Fraction, Fraction, Fraction] = (
                        sum(
                            (
                                coefficient * source_points[vertex][0]
                                for vertex, coefficient in stencil
                            ),
                            Fraction(),
                        ),
                        sum(
                            (
                                coefficient * source_points[vertex][1]
                                for vertex, coefficient in stencil
                            ),
                            Fraction(),
                        ),
                        sum(
                            (
                                coefficient * source_points[vertex][2]
                                for vertex, coefficient in stencil
                            ),
                            Fraction(),
                        ),
                    )
                    if len(stencil) == 1:
                        row = stencil[0][0]
                    else:
                        key = (carrier, grid)
                        row = keys.get(key)
                        if row is None:
                            row = len(points)
                            if row + 1 > limits.maximum_vertices:
                                raise _failure(
                                    "Actual original-source cut vertices exceed original capacity.",
                                    resource=True,
                                )
                            admit_retained(point_bytes)
                            keys[key] = row
                            points.append(point)
                            stencils.append(stencil)
                        elif points[row] != point:
                            raise _failure(
                                "Shared scientific cut vertex has inconsistent exact original source restrictions."
                            )
                    vertex_rows.append(row)
                loops = [
                    tuple(
                        vertex_rows[int(vertex)]
                        for vertex in clipped.face_vertices[
                            int(clipped.face_offsets[face]) : int(
                                clipped.face_offsets[face + 1]
                            )
                        ]
                    )
                    for face in range(len(clipped.face_planes))
                ]
                entries = sum(len(loop) for loop in loops)
                if face_entries + entries > limits.maximum_connectivity_entries:
                    raise _failure(
                        "Actual original-source cut face entries exceed original capacity.",
                        resource=True,
                    )
                admit_retained(512 + 32 * entries)
                face_entries += entries
                if hex_count + len(vertex_rows) > limits.maximum_cells:
                    raise _failure(
                        "Actual cut-corner pure-hex count exceeds original cell capacity.",
                        resource=True,
                    )
                hex_count += len(vertex_rows)
                cut_cells.append(loops)
                cut_parents.append(parent)
        if not cut_cells:
            raise _failure("Exact source-cell clipping has no positive-volume pieces.")
        coordinates = _family_host_array((len(points), 3), np.float64)
        for row, point in enumerate(points):
            coordinates[row] = rounded_point(point)
        added = len(points) - original
        poly = CellMesh.from_polyhedra(
            coordinates,
            cut_cells,
            vertex_global_ids=_vertex_ids(source_mesh, added),
            numeric_version=source_mesh.numeric_version,
        )
        connectivity = poly.connectivity
        if not isinstance(connectivity, PolyhedralConnectivity):
            raise RuntimeError("Cut complex lost polyhedral connectivity.")
        if (
            connectivity.edge_count > limits.maximum_edges
            or connectivity.face_count > limits.maximum_faces
        ):
            raise _failure(
                "Actual scientific cut complex exceeds original edge/face capacities.",
                resource=True,
            )
        offsets, vertices, coefficients = _csr(stencils)
        parents = _family_host_array((len(cut_parents),), np.int64)
        parents[:] = cut_parents
        authority = ExactPlcCellGeometryConvexSource(
            source_mesh,
            source_geometry,
            offsets,
            vertices,
            coefficients,
            poly,
            parent_ids[parents],
        )
        integer = _family_host_array((len(active_grid), 3), np.int64)
        integer[:] = sorted(active_grid)
        address = MortonAddressPlan(
            (0.0, 0.0, 0.0), tuple(float(1 << depth) for _ in range(3)), depth
        )
        codes = morton_encode_integer_host(integer, depth)
        refined = [
            np.unique(codes >> np.uint64(3 * (depth - level))) for level in range(depth)
        ]
        prefixes, levels, _ = refined_octree_leaves(address, refined, balanced=True)
        result = (poly, authority, parents, prefixes, levels, work)
    finally:
        if workspace is not None:
            workspace.set_bound(starting)
    if workspace is not None:
        workspace.retain_owner(result)
    return result


def _canonical_cut_origin(
    mesh: CellMesh,
    points: tuple[tuple[Fraction, Fraction, Fraction], ...],
    base: tuple[Fraction, ...],
    step: Fraction,
    limits: MeshingLimits,
) -> tuple[tuple[Fraction, ...], CutGridOriginProof]:
    """Exclude grid/source coincidences by an exact integer-polynomial bound.

    Origin=base+step*(t,t²,t³). For every nonzero incidence polynomial,
    clearing its original rational denominators gives integer coefficients.
    If 0<t<1/(1+sum(abs(higher coefficients))), its first nonzero term
    dominates all later terms. This holds for every integer grid address;
    no trial clipping, source movement, or field-iteration allowance is used.
    """
    maximum = 0
    work = 0
    execution = current_native_execution_budget()
    constraints = (
        3 * len(points) + 3 * mesh.entity_set(1).count + mesh.entity_set(2).count
    )
    _budget(limits, 0, 0, 0, 32768, 256 * constraints)
    if execution is not None:
        execution.admit_work_bound(256 * constraints)

    def admit(coefficients: Sequence[Fraction]) -> None:
        nonlocal maximum, work
        if execution is not None:
            execution.charge(work=32)
        work += 32
        denominator = lcm(*(value.denominator for value in coefficients))
        integers = tuple(
            value.numerator * (denominator // value.denominator) for value in coefficients
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
            "Balanced cut origin proof requires tetrahedral source connectivity."
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
            constant = direction[y] * (base[x] - a[x]) - direction[x] * (base[y] - a[y])
            admit((constant, *higher))
            edge_count += 1
    faces = _entities(mesh, 2)
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
            raise _failure("An original source SCI face has a deficient exact plane.")
        constant = sum(
            (normal[axis] * (base[axis] - a[axis]) for axis in range(3)), Fraction(0)
        )
        admit((constant, *(step * value for value in normal)))
    parameter = Fraction(1, maximum + 2)
    origin = tuple(base[axis] + step * parameter ** (axis + 1) for axis in range(3))
    proof = CutGridOriginProof(
        base, parameter, maximum, 3 * len(points), edge_count, len(faces), work
    )
    return origin, proof


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
) -> BalancedCutHexConstruction:
    """Construct and independently certify original-source pure cut-grid hexes."""
    if schedule.route != "balanced_grid":
        raise ValueError(
            "Source-cell cut closure belongs to the explicit balanced-grid route."
        )
    if not isinstance(construction, VolumeConstruction) or not isinstance(
        source_geometry.exact_source, ExactPlcCellGeometrySource
    ):
        raise TypeError(
            "Balanced PLC cuts require their genuine retained original PLC construction and coordinate source."
        )
    if complex_.boundary == "fixed":
        lengths = np.diff(np.asarray(complex_.polygon_offsets))
        incompatible = np.flatnonzero(lengths != 4)
        if incompatible.size:
            raise _failure(
                f"Immutable original boundary polygons {tuple(int(row) for row in incompatible)} have nonquadrilateral arity and cannot remain unchanged pure-hex faces."
            )
        raise _failure(
            "The selected cut-corner template subdivides immutable original quadrilateral faces; unchanged-face closure is required."
        )
    if not np.isfinite(target_size) or target_size <= 0:
        raise ValueError("Original cut target size must be finite and positive.")
    if (
        current_native_execution_budget() is None
        or current_native_host_workspace() is None
    ):
        raise _failure(
            "Balanced cut generation requires its original complete request execution/storage scope.",
            resource=True,
        )
    receipts = []
    with _cut_stage("source_and_origin_preparation", None, limits, receipts):
        source_mesh = construction.mesh
        points, cells = _source_corner_bank(source_mesh, source_geometry)
        exponent = floor(log2(target_size)) + 1
        step = Fraction(2) ** exponent
        base = tuple(
            Fraction(floor(min(point[axis] for point in points) / step) - 1) * step
            for axis in range(3)
        )
        origin, origin_proof = _canonical_cut_origin(
            source_mesh, points, base, step, limits
        )
    request = NativeCertificationRequest(
        MeshCertificationSchedule("volume_plc"),
        source_geometry.exact_source.domain_source_id,
        source_geometry.exact_source.domain_source_revision,
        limits,
        domain=construction.domain,
    )
    with (
        _cut_stage("source_clipping", origin, limits, receipts),
        measure_phase(record_phase, "construction"),
    ):
        poly, authority, parents, prefixes, levels, clipping_work = _grid_parts(
            source_mesh, source_geometry, points, cells, origin, step, schedule, limits
        )
    try:
        with (
            _cut_stage("corner_closure", origin, limits, receipts),
            measure_phase(record_phase, "topology_construction"),
        ):
            prepared = prepare_cut_corner_hexes(poly, authority, limits)
    except CutCornerLinkError as error:
        failure = _failure(
            f"Original source-cell link remains incompatible with the selected corner template after exact grid-incidence exclusion: {error.witness}."
        )
        failure.cut_failure_prefix = error.cut_failure_prefix
        raise failure from error
    with (
        _cut_stage("exact_maps_embedding_quality", origin, limits, receipts),
        measure_phase(record_phase, "curving"),
    ):
        actual = realize_cut_corner_hexes(
            prepared, limits, certificate_limits=request.certificate_limits
        )
        require_cut_corner_quality(actual, schedule)
    with _cut_stage("original_scientific_ancestry", origin, limits, receipts):
        original_offsets, original_vertices, _ = compose_original_plc_supports(prepared)
        supports = _family_host_array((actual.mesh.coordinates.shape[0], 4), np.int64)
        supports.fill(-1)
        for row in range(supports.shape[0]):
            indices = original_vertices[original_offsets[row] : original_offsets[row + 1]]
            supports[row, : len(indices)] = indices
        parent_cells = parents[prepared.parent_cells]
        dimensions, rows = _ancestry(source_mesh, actual.mesh, supports)
        source_connectivity = source_mesh.connectivity
        if not isinstance(source_connectivity, TetrahedralConnectivity):
            raise RuntimeError("Balanced cut source lost tetrahedral connectivity.")
        faces = _entities(source_mesh, 2)
        incidence = np.bincount(
            np.asarray(source_connectivity.cell_faces).reshape(-1),
            minlength=len(faces),
        )
        target_faces = _entities(actual.mesh, 2)
        on_boundary = (dimensions[2] == 2) & (
            incidence[np.minimum(rows[2], len(faces) - 1)] == 1
        )
        boundary = _family_host_array((actual.mesh.coordinates.shape[0],), np.bool_)
        boundary.fill(False)
        boundary[target_faces[on_boundary].reshape(-1)] = True
        valences = np.bincount(
            prepared.hexes.reshape(-1), minlength=actual.mesh.coordinates.shape[0]
        ).astype(np.int64)
        singular = np.flatnonzero((~boundary) & (valences != 8)).astype(np.int64)
    from ..geometry._mesh_certificates import certify_domain_coverage

    with (
        _cut_stage("original_domain_coverage", origin, limits, receipts),
        measure_phase(record_phase, "certification"),
    ):
        coverage = certify_domain_coverage(
            actual.mesh,
            actual.geometry,
            construction.domain,
            construction.cell_regions[parent_cells],
            embedding=actual.embedding,
            limits=request.certificate_limits,
        )
        if coverage.status != "certified":
            raise _failure(
                f"Actual cut-grid original source coverage is not certified: {tuple(finding.check for finding in coverage.findings)}."
            )
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
            "origin_proof": origin_proof,
        }
    )
    performed = sum(int(receipt.execution.work_evidence[0]) for receipt in receipts)
    extraction = DualExtraction(
        actual.mesh,
        source_mesh,
        supports,
        parent_cells,
        dimensions,
        rows,
        actual.validity,
        valences,
        boundary,
        singular,
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
        prefixes,
        levels,
        origin,
        step,
        origin_proof,
        actual.embedding,
        coverage,
        tuple(receipts),
        performed,
        identity,
    )
