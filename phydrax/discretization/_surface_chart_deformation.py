#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"Prepare bounded surface-map deformation through material correspondence.\n\nNative rational supporting faces partition retained source material charts.\nPhysical UV/source maps and nonlinear trim-ribbon coverage remain independent;\nno rounded material carrier or sampled map is a completeness condition."

from __future__ import annotations

import math
from fractions import Fraction
from typing import cast

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

import phydrax.discretization._mapped_geometry_measure as mapped_measure_owner
import phydrax.discretization.fem._reference_types as reference_types_owner
from phydrax.discretization._reference_cell_types import CellBlock, CellGeometryElement
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.geometry._rational import fraction_endpoint

from .. import _bvh as imported_bvh, _meshcore
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._host_storage import declared_object_storage_bytes
from .._meshcore import exact_orient2d, NativeBatchTicket, NativeExecutionScope
from .._strict import StrictModule
from .._validation import positive_integer
from ..geometry import _supermesh as supermesh_owner
from ..geometry._domain_boundary import MeshingDomainBoundarySource
from ..geometry._domain_chart_chain import verify_chart_chain
from ..geometry._global_embedding import (
    certify_global_embedding,
    EmbeddingCounts,
    GlobalEmbeddingCertificate,
)
from ..geometry._mesh_certificates import MeshCertificateBinding
from ..geometry._meshing_domain import MeshingDomain, PatchCurveUse, PatchPoleUse
from ..geometry._rational import rational_vector
from ..geometry._source_fidelity import SourceFidelityCertificate
from ..geometry.brep._patches import LineCurve
from ..linalg import _small_batched as imported_linalg__small_batched
from ..typing import checked, ConvertibleToArray
from . import (
    _cell_geometry_transfer as imported_cell_geometry_transfer,
    _chart_execution as chart_execution,
    _chart_reconstruction as chart_reconstruction,
    _coordinate_budget as imported_coordinate_budget,
    _coordinate_enclosure as imported_coordinate_enclosure,
    _surface_chart_records as surface_chart_records,
)
from ._cell_geometry import (
    BarycentricCellGeometryElement,
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    LayerColumnCellGeometryElement,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    require_scalar_coordinate_element,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from ._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from ._cell_mesh import CellMesh
from ._chart_certification import certify_chart_endpoint
from ._chart_geometry import area, merge_oriented_boundary_chain
from ._coordinate_budget import CoordinateEnclosureBudget
from ._coordinate_enclosure import (
    add,
    axes,
    constant,
    coordinate_expressions,
    expression_add,
    expression_bernstein_coefficients,
    expression_evaluate,
    expression_node_count,
    expression_scale,
    Polynomial,
    scale,
)
from ._exact_algebraic_geometry import rational_coefficient


type _FloatArray = NDArray[np.float64]
type _ExactPoint = tuple[Fraction, Fraction]
type _ExactTriangle = tuple[_ExactPoint, _ExactPoint, _ExactPoint]
type _IntArray = NDArray[np.int64]


def _coordinate_elements(
    elements: tuple[CellGeometryElement, ...],
) -> tuple[
    reference_types_owner.FiniteElementSpec
    | BarycentricCellGeometryElement
    | RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement
    | SplineCellGeometryElement
    | LayerColumnCellGeometryElement,
    ...,
]:
    "Resolve the ordered polynomial/rational scalar source owners."

    resolved = []
    for candidate in elements:
        element = require_scalar_coordinate_element(candidate, "Surface coordinates")
        if element.cell_kind != "triangle" or element.topological_dimension != 2:
            raise ValueError(
                "surface_chart.coordinate_elements.pieces must satisfy the material reference contract; got Surface coordinate owners must use triangle reference cells."
            )
        if isinstance(element, reference_types_owner.FiniteElementSpec) and (
            element.conformity != "H1"
            or element.value_shape
            or element.mapping != "identity"
        ):
            raise ValueError(
                "surface_chart.coordinate_elements.binding must satisfy the declared geometry and cell identities; got Surface coordinates require scalar identity-mapped H1 elements."
            )
        resolved.append(element)
    return tuple(resolved)


def _same_periodic_chart_use(
    domain: MeshingDomain,
    patch: int,
    first: tuple[float, ...],
    second: tuple[float, ...],
    /,
) -> bool:
    "Compare only authored deck shifts, never physical coincidences."
    for a, b, period in zip(
        first, second, domain.patches[patch].surface.periods, strict=True
    ):
        difference = Fraction(b) - Fraction(a)
        if not difference:
            continue
        if period is None or (difference / Fraction(period)).denominator != 1:
            return False
    return True


def _witness_order(
    witness: surface_chart_records.SurfaceChartWitness,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: MeshingDomain,
) -> _IntArray:
    if (witness.geometry_id, witness.topology_id, witness.domain_id) != (
        cell_geometry_id(geometry),
        mesh.topology_id,
        domain.domain_id,
    ):
        raise ValueError(
            "surface_chart.witness_order.binding must satisfy the declared geometry and cell identities; got Chart binding is stale for the geometry, topology or domain."
        )
    ids = surface_chart_records.chart_cell_ids(mesh)
    declared = np.asarray(witness.cell_global_ids)
    if not np.array_equal(np.sort(ids), declared):
        raise ValueError(
            "surface_chart.witness_order.binding must satisfy the declared geometry and cell identities; got Chart binding must cover every scientific physical cell ID exactly once."
        )
    patches = np.asarray(witness.patches)
    if np.any(patches >= len(domain.patches)) or any(
        entity != domain.entity_id(2, int(patch))
        or path != domain.source_occurrences[2][int(patch)]
        for entity, path, patch in zip(
            witness.geometry_entity_ids, witness.occurrence_paths, patches, strict=True
        )
    ):
        raise ValueError(
            "surface_chart.witness_order.binding must satisfy the declared geometry and cell identities; got Chart binding has stale ordered geometry entities or occurrence paths."
        )
    order = np.asarray(np.searchsorted(declared, ids), dtype=np.int64)
    vertex_charts: dict[tuple[int, int], tuple[float, float]] = {}
    chart_vertices: dict[tuple[int, tuple[float, float]], int] = {}
    offset = 0
    for block in mesh.blocks:
        for vertices in np.asarray(block.vertices):
            witness_row = int(order[offset])
            patch = int(patches[witness_row])
            for vertex, chart in zip(
                vertices, np.asarray(witness.charts)[witness_row], strict=True
            ):
                key = patch, int(np.asarray(mesh.vertex_global_ids)[int(vertex)])
                value = (float(chart[0]), float(chart[1]))
                if key in vertex_charts and not _same_periodic_chart_use(
                    domain, patch, vertex_charts[key], value
                ):
                    raise ValueError(
                        "surface_chart.witness_order.pieces must satisfy the material reference contract; got Shared physical vertices disagree in the declared occurrence chart."
                    )
                vertex_charts[key] = value
                chart_key = patch, value
                if chart_key in chart_vertices and chart_vertices[chart_key] != key[1]:
                    raise ValueError(
                        "surface_chart.witness_order.pieces must satisfy the material reference contract; got One occurrence chart vertex cannot merge distinct scientific physical vertices."
                    )
                chart_vertices[chart_key] = key[1]
            offset += 1
    return order


def _carrier(
    witness: surface_chart_records.SurfaceChartWitness,
    physical_order: _IntArray,
    patch: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[CellMesh, _IntArray, _FloatArray]:
    selected = np.flatnonzero(np.asarray(witness.patches) == patch)
    rows_by_witness = np.empty(physical_order.size, dtype=np.int64)
    rows_by_witness[physical_order] = np.arange(physical_order.size, dtype=np.int64)
    charts = np.asarray(witness.charts)[selected]

    signs = exact_orient2d(charts[:, 0], charts[:, 1], charts[:, 2])
    if np.any(signs == 0):
        raise ValueError(
            "surface_chart.carrier.pieces must satisfy the material reference contract; got A surface chart must be a nondegenerate UV triangle."
        )
    permutations = np.tile(np.arange(3, dtype=np.int32), (selected.size, 1))
    permutations[signs < 0] = np.asarray([0, 2, 1], dtype=np.int32)
    ordered = np.take_along_axis(charts, permutations[..., None], axis=1)
    points, inverse = np.unique(ordered.reshape((-1, 2)), axis=0, return_inverse=True)
    mesh = CellMesh.from_triangles(
        points,
        inverse.reshape((-1, 3)),
        cell_global_ids=np.asarray(witness.cell_global_ids)[selected],
    )
    reference = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)[
        permutations
    ]
    return mesh, rows_by_witness[selected], reference


def _material_reference_arguments(
    vertices: _ExactTriangle, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...]:
    variables = axes(2, budget=budget)
    return tuple(
        add(
            constant(vertices[0][axis], 2, budget=budget),
            add(
                scale(variables[0], vertices[1][axis] - vertices[0][axis], budget=budget),
                scale(variables[1], vertices[2][axis] - vertices[0][axis], budget=budget),
                budget=budget,
            ),
            budget=budget,
        )
        for axis in range(2)
    )


def _exact_retained_bytes(value: object) -> int:
    "Bound material metadata by structure and integer bit lengths."
    size = declared_object_storage_bytes(value)
    match value:
        case Fraction():
            return (
                size
                + declared_object_storage_bytes(value.numerator)
                + declared_object_storage_bytes(value.denominator)
            )
        case tuple():
            return size + sum(_exact_retained_bytes(item) for item in value)
        case surface_chart_records.PreparedSurfaceChartPiece():
            return size + _exact_retained_bytes(
                (
                    value.source_row,
                    value.target_row,
                    value.source_cell_global_id,
                    value.target_cell_global_id,
                    value.exact_source_reference_vertices,
                    value.exact_target_reference_vertices,
                    value.source_reference_area,
                    value.target_reference_area,
                    value.native_boundary_labels,
                    value.exact_native_source_polygon,
                    value.native_polygon_vertex_indices,
                    value.piece_id,
                )
            )
        case _:
            return size


def _native_material_candidates(
    source: tuple[_ExactTriangle, ...],
    target: tuple[_ExactTriangle, ...],
    maximum_pairs: int,
    maximum_memory: int,
) -> tuple[_IntArray, _IntArray]:

    boxes = []
    for charts in (source, target):
        lower = np.asarray(
            [
                [
                    fraction_endpoint(
                        min(point[axis] for point in chart), upper=False, saturate=True
                    )
                    for axis in range(2)
                ]
                for chart in charts
            ]
        )
        upper = np.asarray(
            [
                [
                    fraction_endpoint(
                        max(point[axis] for point in chart), upper=True, saturate=True
                    )
                    for axis in range(2)
                ]
                for chart in charts
            ]
        )
        boxes.append(imported_bvh.prepare_bvh(lower, upper, dtype=jnp.float64))
    left, right, count = [], [], 0
    for target_rows, source_rows in imported_bvh.bvh_overlap_pair_blocks(
        boxes[1], boxes[0]
    ):
        count += source_rows.size
        if count > maximum_pairs or count * 32 > maximum_memory:
            raise surface_chart_records.SurfaceChartResourceError(
                "surface_chart.native_material_candidates.budget must satisfy the admitted resource limits; got material candidate routing exceeds the source pair/storage allowance.",
                (
                    ("material_candidate_pairs", count, maximum_pairs),
                    ("material_candidate_bytes", count * 32, maximum_memory),
                ),
            )
        left.append(np.asarray(source_rows, dtype=np.int64))
        right.append(np.asarray(target_rows, dtype=np.int64))
    if not left:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(left), np.concatenate(right)


def _prepare_contained_material_pieces(
    source: tuple[_ExactTriangle, ...],
    target: tuple[_ExactTriangle, ...],
    source_ids: _IntArray,
    target_ids: _IntArray,
    maximum_pairs: int,
    maximum_work: int,
    maximum_memory: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> (
    tuple[tuple[surface_chart_records.PreparedSurfaceChartPiece, ...], int, int, int]
    | None
):
    "Construct an whole-child partition already proved by chart closure."
    candidates = _native_material_candidates(
        source, target, maximum_pairs, maximum_memory
    )
    by_target: dict[int, list[int]] = {}
    for old, new in zip(*candidates, strict=True):
        by_target.setdefault(int(new), []).append(int(old))
    pieces = []
    source_areas = [Fraction(0)] * len(source)
    target_areas = [Fraction(0)] * len(target)
    retained = candidates[0].nbytes + candidates[1].nbytes
    work = 0
    references: _ExactTriangle = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    for new in range(len(target)):
        selected: tuple[int, _ExactTriangle] | None = None
        for old in sorted(by_target.get(new, ()), key=lambda row: int(source_ids[row])):
            old_vertices = cast(
                _ExactTriangle,
                tuple(
                    surface_chart_records.material_reference(source[old], point)
                    for point in target[new]
                ),
            )
            work += 18
            if area(old_vertices) > 0 and all(
                min(u, v, 1 - u - v) >= 0 for u, v in old_vertices
            ):
                selected = old, old_vertices
                break
        if selected is None:
            return None
        old, old_vertices = selected
        mapped = cast(
            _ExactTriangle,
            tuple(
                surface_chart_records.material_reference(target[new], point)
                for point in source[old]
            ),
        )
        origin = mapped[0]
        first = (
            mapped[1][0] - origin[0],
            mapped[1][1] - origin[1],
        )
        second = (
            mapped[2][0] - origin[0],
            mapped[2][1] - origin[1],
        )
        planes = (
            (Fraction(-1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(-1), Fraction(0)),
            (Fraction(1), Fraction(1), Fraction(1)),
            (-first[0], -second[0], origin[0]),
            (-first[1], -second[1], origin[1]),
            (
                first[0] + first[1],
                second[0] + second[1],
                1 - origin[0] - origin[1],
            ),
        )
        labels = []
        for point, following in zip(
            old_vertices, old_vertices[1:] + old_vertices[:1], strict=True
        ):
            supported = [
                index
                for index, (a, b, c) in enumerate(planes)
                if a * point[0] + b * point[1] == c
                and a * following[0] + b * following[1] == c
            ]
            if not supported:
                return None
            labels.append(min(supported))
        new_vertices = tuple(
            surface_chart_records.material_point(mapped, point) for point in old_vertices
        )
        if new_vertices != references:
            return None
        old_area = area(old_vertices)
        new_area = Fraction(1, 2)
        labels_ = tuple(labels)
        identity = surface_chart_records.material_piece_id(
            int(source_ids[old]),
            int(target_ids[new]),
            old_vertices,
            references,
            old_vertices,
            labels_,
            (0, 1, 2),
        )
        piece = surface_chart_records.PreparedSurfaceChartPiece(
            source_row=old,
            target_row=new,
            source_cell_global_id=int(source_ids[old]),
            target_cell_global_id=int(target_ids[new]),
            exact_source_reference_vertices=old_vertices,
            exact_target_reference_vertices=references,
            source_reference_area=old_area,
            target_reference_area=new_area,
            native_boundary_labels=labels_,
            piece_id=identity,
            exact_native_source_polygon=old_vertices,
            native_polygon_vertex_indices=(0, 1, 2),
        )
        retained += _exact_retained_bytes(piece)
        pieces.append(piece)
        source_areas[old] += old_area
        target_areas[new] += new_area
        work += 64
    if any(area != Fraction(1, 2) for area in (*source_areas, *target_areas)):
        return None
    if work > maximum_work or retained > maximum_memory:
        raise surface_chart_records.SurfaceChartResourceError(
            "surface_chart.prepare_contained_material_pieces.budget must satisfy the admitted resource limits; got Contained material partition exceeds source resources.",
            (
                ("material_work", work, maximum_work),
                ("material_bytes", retained, maximum_memory),
            ),
        )
    budget.reserve(work, retained)
    return tuple(pieces), work, candidates[0].size, retained


class _MaterialPolygonConstruction(StrictModule):
    polygon: tuple[_ExactPoint, ...]


def _construct_material_polygon(
    supports: NDArray[np.int32],
    labels: NDArray[np.int32],
    planes: tuple[tuple[Fraction, Fraction, Fraction], ...],
    budget: CoordinateEnclosureBudget,
) -> _MaterialPolygonConstruction:
    "Construct and authenticate the native rational polygon supports."
    polygon_points: list[_ExactPoint] = []
    for left, right in supports:
        a, b = planes[int(left)], planes[int(right)]
        construction = imported_linalg__small_batched.prepare_exact_small_linear_actions(
            ((a[0], a[1]), (b[0], b[1])),
            ((a[2],), (b[2],)),
            budget=budget,
        )
        if construction.actions is None:
            raise ValueError(
                "surface_chart.construct_material_polygon.reference_map must satisfy the regular reference-map contract; got Native material support has no nonsingular source construction."
            )
        point = construction.actions[0][0], construction.actions[1][0]
        if any(x * point[0] + y * point[1] > z for x, y, z in planes):
            raise ValueError(
                "surface_chart.construct_material_polygon.pieces must satisfy the material reference contract; got Native material support violates an source halfplane."
            )
        polygon_points.append(point)
    polygon = tuple(polygon_points)
    if len(set(polygon)) != len(polygon) or area(polygon) <= 0:
        raise ValueError(
            "surface_chart.construct_material_polygon.pieces must satisfy the material reference contract; got Native material polygon lacks a positive distinct-vertex partition."
        )
    for slot, label in enumerate(labels):
        a, b, c = planes[int(label)]
        if any(
            a * point[0] + b * point[1] != c
            for point in (polygon[slot], polygon[(slot + 1) % len(polygon)])
        ):
            raise ValueError(
                "surface_chart.construct_material_polygon.pieces must satisfy the material reference contract; got Native material edge changes its source support."
            )
    return _MaterialPolygonConstruction(polygon=polygon)


@checked
def prepare_exact_material_pieces(
    source: tuple[_ExactTriangle, ...],
    target: tuple[_ExactTriangle, ...],
    source_ids: _IntArray,
    target_ids: _IntArray,
    maximum_pairs: int,
    maximum_work: int,
    maximum_memory: int,
    /,
    *,
    partitions_certified: bool = False,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[surface_chart_records.PreparedSurfaceChartPiece, ...], int, int, int]:
    "Native halfplane topology plus coefficient construction."

    before = budget.work_units
    if maximum_work <= 0 or maximum_memory <= 0 or maximum_pairs <= 0:
        raise surface_chart_records.SurfaceChartResourceError(
            "surface_chart.prepare_exact_material_pieces.budget must satisfy the admitted resource limits; got No source material overlap resources remain.",
            (("material_work", 1, maximum_work), ("material_bytes", 1, maximum_memory)),
        )
    if partitions_certified:
        contained = _prepare_contained_material_pieces(
            source,
            target,
            source_ids,
            target_ids,
            maximum_pairs,
            maximum_work,
            maximum_memory,
            budget=budget,
        )
        if contained is not None:
            return contained
    embedding_work, embedding_pairs = 0, 0
    if source != target and not partitions_certified:
        for charts, ids in ((source, source_ids), (target, target_ids)):
            embedding = prepare_exact_material_pieces(
                charts,
                charts,
                ids,
                ids,
                maximum_pairs - embedding_pairs,
                maximum_work - embedding_work,
                maximum_memory,
                budget=budget,
            )
            embedding_work += embedding[1]
            embedding_pairs += embedding[2]
            del embedding
    before = budget.work_units
    native_work = 0
    pieces = []
    source_areas, target_areas = [Fraction(0)] * len(source), [Fraction(0)] * len(target)
    candidates = _native_material_candidates(
        source, target, maximum_pairs - embedding_pairs, maximum_memory
    )
    retained = candidates[0].nbytes + candidates[1].nbytes
    try:
        with budget.measurement():
            with budget.activate():
                budget.reserve(0, retained)
                for old_value, new_value in zip(*candidates, strict=True):
                    old, new = int(old_value), int(new_value)
                    budget.reserve(64)
                    mapped = (
                        surface_chart_records.material_reference(
                            target[new], source[old][0]
                        ),
                        surface_chart_records.material_reference(
                            target[new], source[old][1]
                        ),
                        surface_chart_records.material_reference(
                            target[new], source[old][2]
                        ),
                    )
                    origin = mapped[0]
                    first = tuple(mapped[1][axis] - origin[axis] for axis in range(2))
                    second = tuple(mapped[2][axis] - origin[axis] for axis in range(2))
                    planes = (
                        (Fraction(-1), Fraction(0), Fraction(0)),
                        (Fraction(0), Fraction(-1), Fraction(0)),
                        (Fraction(1), Fraction(1), Fraction(1)),
                        (-first[0], -second[0], origin[0]),
                        (-first[1], -second[1], origin[1]),
                        (
                            first[0] + first[1],
                            second[0] + second[1],
                            1 - origin[0] - origin[1],
                        ),
                    )
                    remaining = (
                        maximum_work
                        - embedding_work
                        - (budget.work_units - before)
                        - native_work
                    )
                    if remaining <= 0 or maximum_memory - retained <= 0:
                        raise surface_chart_records.SurfaceChartResourceError(
                            "surface_chart.prepare_exact_material_pieces.budget must satisfy the admitted resource limits; got native material clipping has no remaining resources.",
                            (
                                (
                                    "material_work",
                                    embedding_work
                                    + budget.work_units
                                    - before
                                    + native_work
                                    + 1,
                                    maximum_work,
                                ),
                                ("material_bytes", retained + 1, maximum_memory),
                            ),
                        )

                    supports, labels, clipping_work = (
                        _meshcore.clip_reference_triangle_exact(
                            planes,
                            maximum_scratch_bytes=maximum_memory - retained,
                            maximum_work_units=remaining,
                        )
                    )
                    native_work += clipping_work
                    if supports.shape[0] < 3:
                        continue
                    _phase = _construct_material_polygon(
                        supports=supports, labels=labels, planes=planes, budget=budget
                    )
                    polygon = _phase.polygon
                    for slot in range(1, len(polygon) - 1):
                        old_vertices = polygon[0], polygon[slot], polygon[slot + 1]
                        new_vertices = (
                            surface_chart_records.material_point(mapped, old_vertices[0]),
                            surface_chart_records.material_point(mapped, old_vertices[1]),
                            surface_chart_records.material_point(mapped, old_vertices[2]),
                        )
                        old_area, new_area = (
                            abs(area(old_vertices)),
                            abs(area(new_vertices)),
                        )
                        if not old_area or not new_area:
                            raise ValueError(
                                "surface_chart.prepare_exact_material_pieces.pieces must satisfy the material reference contract; got A material piece is collapsed."
                            )
                        identity = surface_chart_records.material_piece_id(
                            int(source_ids[old]),
                            int(target_ids[new]),
                            old_vertices,
                            new_vertices,
                            polygon,
                            tuple(map(int, labels)),
                            (0, slot, slot + 1),
                        )
                        piece = surface_chart_records.PreparedSurfaceChartPiece(
                            source_row=old,
                            target_row=new,
                            source_cell_global_id=int(source_ids[old]),
                            target_cell_global_id=int(target_ids[new]),
                            exact_source_reference_vertices=old_vertices,
                            exact_target_reference_vertices=new_vertices,
                            source_reference_area=old_area,
                            target_reference_area=new_area,
                            native_boundary_labels=tuple(map(int, labels)),
                            piece_id=identity,
                            exact_native_source_polygon=polygon,
                            native_polygon_vertex_indices=(0, slot, slot + 1),
                        )
                        size = _exact_retained_bytes(piece)
                        budget.reserve(24, size)
                        retained += size
                        pieces.append(piece)
                        source_areas[old] += old_area
                        target_areas[new] += new_area
    except imported_coordinate_budget.CoordinateEnclosureResourceError as error:
        raise surface_chart_records.SurfaceChartResourceError(
            "surface_chart.prepare_exact_material_pieces.budget must satisfy the admitted resource limits; got native material coefficient construction exhausts its source budget.",
            ((error.resource, int(error.requested), int(error.limit)),),
        ) from error
    if any(area != Fraction(1, 2) for area in (*source_areas, *target_areas)):
        raise ValueError(
            "surface_chart.prepare_exact_material_pieces.pieces must satisfy the material reference contract; got Native material overlap has a gap or double cover; whole source and target partitions required."
        )
    work = embedding_work + budget.work_units - before + native_work
    if work > maximum_work:
        raise surface_chart_records.SurfaceChartResourceError(
            "surface_chart.prepare_exact_material_pieces.budget must satisfy the admitted resource limits; got material clipping exceeds its aggregate source work allowance.",
            (("material_work", work, maximum_work),),
        )
    return tuple(pieces), work, candidates[0].size + embedding_pairs, retained


def _material_displacement_bounds(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    source_rows: _IntArray,
    target_rows: _IntArray,
    pieces: tuple[surface_chart_records.PreparedSurfaceChartPiece, ...],
    maximum_work: int,
    maximum_memory: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[_FloatArray, int]:
    "Continuous source-map difference on every native material piece."

    before = budget.work_units
    if maximum_work <= 0 or maximum_memory <= 0:
        raise surface_chart_records.SurfaceChartResourceError(
            "surface_chart.material_displacement_bounds.budget must satisfy the admitted resource limits; got Material-map difference has no remaining source coefficient resources.",
            (
                ("material_difference_work", 1, maximum_work),
                ("material_difference_bytes", 1, maximum_memory),
            ),
        )
    maps = []
    try:
        with budget.measurement():
            with budget.activate():
                for mesh, geometry, rows in (
                    (source_mesh, source_geometry, source_rows),
                    (target_mesh, target_geometry, target_rows),
                ):
                    elements, routes, _ = geometry.resolve(mesh, budget=budget)
                    controls = geometry.source_coordinates(budget=budget)
                    needed = set(map(int, rows))
                    expressions = {}
                    physical_row = 0
                    for element, route in zip(elements, routes, strict=True):
                        for row in np.asarray(route):
                            if physical_row in needed:
                                values = coordinate_expressions(
                                    element,
                                    tuple(controls[int(index)] for index in row),
                                    budget=budget,
                                )
                                if values is None:
                                    raise ValueError(
                                        "surface_chart.material_displacement_bounds.pieces must satisfy the material reference contract; got Material correspondence requires coordinate source expressions."
                                    )
                                expressions[physical_row] = values
                            physical_row += 1
                    maps.append(expressions)
                bounds = []
                for piece in pieces:
                    left = _material_reference_arguments(
                        piece.exact_source_reference_vertices, budget=budget
                    )
                    right = _material_reference_arguments(
                        piece.exact_target_reference_vertices, budget=budget
                    )
                    bound = Fraction(0)
                    for old, new in zip(
                        maps[0][int(source_rows[piece.source_row])],
                        maps[1][int(target_rows[piece.target_row])],
                        strict=True,
                    ):
                        difference = expression_add(
                            imported_coordinate_enclosure.expression_compose(
                                old, left, budget=budget
                            ),
                            expression_scale(
                                imported_coordinate_enclosure.expression_compose(
                                    new, right, budget=budget
                                ),
                                -1,
                                budget=budget,
                            ),
                            budget=budget,
                        )
                        bound += max(
                            abs(value)
                            for value in expression_bernstein_coefficients(
                                difference, "simplex", 2, budget=budget
                            )
                        )
                    bounds.append(
                        fraction_endpoint(
                            rational_coefficient(
                                bound,
                                role="surface_chart.material_displacement_bounds.bound",
                            ),
                            upper=True,
                            saturate=True,
                        )
                    )
    except imported_coordinate_budget.CoordinateEnclosureResourceError as error:
        raise surface_chart_records.SurfaceChartResourceError(
            "surface_chart.material_displacement_bounds.budget must satisfy the admitted resource limits; got Continuous material-map difference exhausts its source coefficient budget.",
            ((error.resource, int(error.requested), int(error.limit)),),
        ) from error
    return np.asarray(bounds, dtype=np.float64), budget.work_units - before


def _source_owned_domain_cover(
    domain: MeshingDomain,
    patch: int,
    carrier: CellMesh,
    source: MeshingDomainBoundarySource,
    maximum_pairs: int,
    maximum_memory_bytes: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    native_ticket: NativeBatchTicket,
) -> int:
    "Retain the source trim ribbons and prove equality of their UV carrier."
    if source.domain.domain_id != domain.domain_id:
        raise ValueError(
            "surface_chart.source_owned_domain_cover.pieces must satisfy the material reference contract; got The source chart cover changes its authored trim source contract."
        )
    records = [record for record in source.chart_triangulations if record[0] == patch]
    if len(records) != 1:
        raise ValueError(
            "surface_chart.source_owned_domain_cover.pieces must satisfy the material reference contract; got The source occurrence requires exactly one chart triangulation."
        )
    (
        _,
        charts,
        points,
        cells,
        boundary,
        provenance,
        restriction_required,
        restriction_vertices,
        restriction_edges,
        restriction_parameters,
    ) = records[0]
    findings, counts = [], []

    def reserve_queries(count: int, /) -> None:
        native_ticket.consume(geometry_queries=count)

    valid, _, _, _ = verify_chart_chain(
        domain,
        patch,
        charts,
        points,
        cells,
        boundary,
        provenance,
        restriction_required,
        restriction_vertices,
        restriction_edges,
        restriction_parameters,
        findings,
        counts,
        maximum_pairs=maximum_pairs,
        reserve_queries=reserve_queries,
        topology_only=True,
        budget=budget,
    )

    native_ticket.consume(sum(completed for _, completed, _ in counts))
    if any(
        finding.check in ("source_trim_pair_capacity", "source_trim_arc_capacity")
        for finding in findings
    ):
        raise surface_chart_records.SurfaceChartResourceError(
            "surface_chart.source_owned_domain_cover.budget must satisfy the admitted resource limits; got The source trim proof exhausts its authored resources.",
            tuple(counts),
        )
    if not valid:
        raise ValueError(
            "surface_chart.source_owned_domain_cover.pieces must satisfy the material reference contract; got The source rooted trim chart cover is not certified."
        )
    original = CellMesh.from_triangles(charts, cells)
    overlap = supermesh_owner.prepare_common_refinement(
        original,
        carrier,
        policy=supermesh_owner.CommonRefinementPolicy(
            coverage=supermesh_owner.CommonRefinementCoverage.COMPLETE,
            maximum_candidate_pairs=maximum_pairs,
            maximum_accepted_pairs=maximum_pairs,
            maximum_memory_bytes=maximum_memory_bytes,
            overlap_simplices=True,
        ),
        budget=budget,
    )
    if overlap.status is supermesh_owner.CommonRefinementStatus.RESOURCE_LIMIT:
        raise surface_chart_records.SurfaceChartResourceError(
            overlap.evidence.reason,
            (
                (
                    "source_trim_overlap_pairs",
                    overlap.evidence.candidate_pair_count,
                    maximum_pairs,
                ),
                (
                    "source_trim_overlap_bytes",
                    overlap.evidence.working_bytes,
                    maximum_memory_bytes,
                ),
            ),
        )
    if not overlap.succeeded:
        raise ValueError(
            f"surface_chart.source_owned_domain_cover.pieces must satisfy the material reference contract; got Source-owned trim carrier coverage refused:{overlap.status.name}.."
        )
    return overlap.evidence.candidate_pair_count


class _AuthoredChartTrimLine(StrictModule):
    origin: tuple[Fraction, ...]
    direction: tuple[Fraction, ...]
    first_parameter: float
    last_parameter: float


def _authored_chart_trim_line(
    use: PatchCurveUse | PatchPoleUse,
) -> _AuthoredChartTrimLine:
    "Resolve a supported authored trim into a rational parameterized line."
    match use:
        case PatchPoleUse():
            origin = rational_vector(use.start)
            direction = tuple(
                Fraction(float(end)) - first
                for first, end in zip(origin, use.end, strict=True)
            )
            first_parameter, last_parameter = 0.0, 1.0
        case PatchCurveUse(pcurve=LineCurve() as pcurve) if (
            not any(
                root is not None
                for root in (
                    use.first_root,
                    use.last_root,
                    use.start_vertex_root,
                    use.end_vertex_root,
                )
            )
            and use.trim_curve is None
        ):
            origin = rational_vector(np.asarray(pcurve.origin))
            direction = rational_vector(np.asarray(pcurve.direction))
            first_parameter, last_parameter = use.first, use.last
        case _:
            raise ValueError(
                "surface_chart.authored_chart_trim_line.pieces must satisfy the material reference contract; got General rooted trims require their source chart-cover owner."
            )
    return _AuthoredChartTrimLine(
        origin=origin,
        direction=direction,
        first_parameter=first_parameter,
        last_parameter=last_parameter,
    )


class _ChartBoundaryLineEntries(StrictModule):
    entries: list[tuple[list[Fraction], tuple[int, int]]]


def _chart_boundary_line_entries(
    remaining: dict[tuple[int, int], int],
    points: _FloatArray,
    origin: tuple[Fraction, ...],
    direction: tuple[Fraction, ...],
    axis: int,
    first_parameter: float,
    last_parameter: float,
) -> _ChartBoundaryLineEntries:
    "Match oriented boundary edges to one authored trim interval."
    entries = []
    for edge, count in remaining.items():
        parameters = []
        for vertex in edge:
            point = rational_vector(points[vertex])
            parameter = (point[axis] - origin[axis]) / direction[axis]
            if any(point[k] != origin[k] + parameter * direction[k] for k in range(2)):
                break
            parameters.append(parameter)
        if (
            len(parameters) == 2
            and min(Fraction(first_parameter), Fraction(last_parameter))
            <= min(parameters)
            and max(parameters)
            <= max(Fraction(first_parameter), Fraction(last_parameter))
            and (parameters[1] - parameters[0])
            * (Fraction(last_parameter) - Fraction(first_parameter))
            > 0
        ):
            if count != 1:
                raise ValueError(
                    "surface_chart.domain_cover.pieces must satisfy the material reference contract; got Chart boundary is multiply covered."
                )
            entries.append((parameters, edge))
    entries.sort(key=lambda item: item[0][0], reverse=last_parameter < first_parameter)
    return _ChartBoundaryLineEntries(entries=entries)


def _domain_cover(
    domain: MeshingDomain,
    patch: int,
    carrier: CellMesh,
    *,
    budget: CoordinateEnclosureBudget,
    native_ticket: NativeBatchTicket,
) -> None:
    "Infer no trims: prove the chain against authored line uses."
    surface = domain.patches[patch].surface
    points = np.asarray(carrier.coordinates)
    box = np.stack((np.min(points, axis=0), np.max(points, axis=0)))
    surface.validate_parameter_box(box)
    # Seam copies and collapsed sides remain distinct chart witnesses.
    # The oriented source chain below establishes coverage.
    cells = np.asarray(carrier.blocks[0].vertices)
    chain: dict[tuple[int, int], int] = {}
    for first, second in np.concatenate(
        (cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]])
    ):
        edge, reverse = (int(first), int(second)), (int(second), int(first))
        if chain.get(reverse, 0):
            chain[reverse] -= 1
        else:
            chain[edge] = chain.get(edge, 0) + 1
    remaining = {edge: count for edge, count in chain.items() if count}
    boundary: list[tuple[int, int]] = []
    provenance: list[tuple[int, int, float, float]] = []
    for loop_index, loop in enumerate(domain.patches[patch].loops):
        for use_index, use in enumerate(loop):
            _phase = _authored_chart_trim_line(use=use)
            origin = _phase.origin
            direction = _phase.direction
            first_parameter = _phase.first_parameter
            last_parameter = _phase.last_parameter
            axis = next((axis for axis, value in enumerate(direction) if value), None)
            if axis is None:
                raise ValueError(
                    "surface_chart.domain_cover.pieces must satisfy the material reference contract; got A chart trim line must be nondegenerate."
                )
            _phase = _chart_boundary_line_entries(
                remaining=remaining,
                points=points,
                origin=origin,
                direction=direction,
                axis=axis,
                first_parameter=first_parameter,
                last_parameter=last_parameter,
            )
            entries = _phase.entries
            for parameters, edge in entries:
                if any(Fraction(float(value)) != value for value in parameters):
                    raise ValueError(
                        "surface_chart.domain_cover.pieces must satisfy the material reference contract; got chart trim parameters are not representable by the coverage owner."
                    )
                boundary.append(edge)
                provenance.append(
                    (loop_index, use_index, float(parameters[0]), float(parameters[1]))
                )
    physical = domain.evaluate(np.full(points.shape[0], patch, dtype=np.int32), points)
    findings, counts = [], []
    required = np.zeros((points.shape[0],), dtype=np.bool_)

    def reserve_queries(count: int, /) -> None:
        native_ticket.consume(geometry_queries=count)

    valid, _, _, _ = verify_chart_chain(
        domain,
        patch,
        points,
        physical,
        cells,
        np.asarray(boundary, dtype=np.int32).reshape((-1, 2)),
        np.asarray(provenance, dtype=np.float64).reshape((-1, 4)),
        required,
        np.empty((0,), dtype=np.int64),
        np.empty((0, 2), dtype=np.int64),
        np.empty((0, 2), dtype=np.int64),
        findings,
        counts,
        reserve_queries=reserve_queries,
        budget=budget,
    )
    if not valid or findings:
        raise ValueError(
            "surface_chart.domain_cover.pieces must satisfy the material reference contract; got Charts do not completely cover the source trim domain."
        )


def _exact_chart_partition_proof(
    source: surface_chart_records.SurfaceChartWitness,
    target: surface_chart_records.SurfaceChartWitness,
    /,
) -> int:
    "Prove the target positive chart complex has the source boundary."

    def records(
        witness: surface_chart_records.SurfaceChartWitness, /
    ) -> tuple[
        dict[int, dict[tuple[_ExactPoint, _ExactPoint], int]],
        dict[int, Fraction],
        int,
    ]:
        charts = surface_chart_records.exact_chart_bank(
            witness.charts, witness.cell_global_ids.shape[0]
        )
        boundaries: dict[int, dict[tuple[_ExactPoint, _ExactPoint], int]] = {}
        areas: dict[int, Fraction] = {}
        work = 0
        for patch, triangle in zip(np.asarray(witness.patches), charts, strict=True):
            signed_area = area(triangle)
            if signed_area <= 0:
                raise ValueError(
                    "surface_chart.records.pieces must satisfy the material reference contract; got Restricted chart partition has a nonpositive cell."
                )
            patch_ = int(patch)
            areas[patch_] = areas.get(patch_, Fraction(0)) + signed_area
            boundary = boundaries.setdefault(patch_, {})
            for first, second in zip(triangle, triangle[1:] + triangle[:1], strict=True):
                reverse = second, first
                if boundary.get(reverse, 0):
                    boundary[reverse] -= 1
                    if boundary[reverse] == 0:
                        del boundary[reverse]
                else:
                    edge = first, second
                    boundary[edge] = boundary.get(edge, 0) + 1
                work += 1
        return (
            {
                patch: merge_oriented_boundary_chain(edges)
                for patch, edges in boundaries.items()
            },
            areas,
            work,
        )

    source_boundary, source_area, source_work = records(source)
    target_boundary, target_area, target_work = records(target)
    if (
        source_boundary != target_boundary
        or source_area != target_area
        or source.domain_id != target.domain_id
    ):
        raise ValueError(
            "surface_chart.exact_chart_partition_proof.pieces must satisfy the material reference contract; got Target restricted charts do not exactly partition the source chart complex."
        )
    return source_work + target_work


def _rational_sqrt_enclosure(value: Fraction, /) -> tuple[Fraction, Fraction]:
    "Tight radical bounds without renewing the coefficient ledger."
    if value <= 0:
        raise ValueError(
            "surface_chart.rational_sqrt_enclosure.measures must satisfy the positive physical measure contract; got A measure radical requires a positive value."
        )
    exponent = (value.numerator.bit_length() - value.denominator.bit_length() + 1) // 2
    upper = Fraction(2) ** exponent
    while upper * upper < value:
        upper *= 2
    while (upper / 2) * (upper / 2) >= value:
        upper /= 2
    for _ in range(8):
        upper = (upper + value / upper) / 2
    return value / upper, upper


def _validity_measure_enclosures(
    validity: CellValidityCertificate, /
) -> tuple[np.ndarray, np.ndarray]:
    """Positive physical triangle-area enclosures from whole-cell Gram bounds."""
    measures = []
    errors = []
    values = tuple(
        zip(
            np.asarray(validity.determinant_lower),
            np.asarray(validity.determinant_upper),
            strict=True,
        )
    )
    for lower, upper in values:
        if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0:
            raise ValueError(
                "surface_chart.validity_measure_enclosures.validity must satisfy the positive cell validity contract; got Validity evidence lacks a positive finite Gram bound."
            )
        # rational Newton bounds consume the determinant certificate
        # without reopening its coefficient-work ledger.
        low = _rational_sqrt_enclosure(Fraction(float(lower)))[0] / 2
        high = _rational_sqrt_enclosure(Fraction(float(upper)))[1] / 2
        midpoint = (low + high) / 2
        value = float(midpoint)
        error = (high - low) / 2 + abs(Fraction(value) - midpoint)
        measures.append(value)
        errors.append(
            fraction_endpoint(error, upper=True, saturate=True) if error else 0.0
        )
    return (
        np.asarray(measures, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
    )


def _restriction_scale(
    source_chart: np.ndarray, target_chart: np.ndarray, /
) -> Fraction | None:
    source = tuple(rational_vector(point) for point in source_chart)
    target = tuple(rational_vector(point) for point in target_chart)
    a, b, c = source
    determinant = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if determinant <= 0:
        return None
    references = []
    for point in target:
        delta = point[0] - a[0], point[1] - a[1]
        u = (delta[0] * (c[1] - a[1]) - delta[1] * (c[0] - a[0])) / determinant
        v = ((b[0] - a[0]) * delta[1] - (b[1] - a[1]) * delta[0]) / determinant
        if min(u, v, 1 - u - v) < 0:
            return None
        references.append((u, v))
    scale = (references[1][0] - references[0][0]) * (
        references[2][1] - references[0][1]
    ) - (references[1][1] - references[0][1]) * (references[2][0] - references[0][0])
    return scale if scale > 0 else None


class _BornSurfaceValidity(StrictModule):
    changed_validity: CellValidityCertificate | None


def _certify_born_surface_cells(
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    changed_masks: list[np.ndarray],
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
) -> _BornSurfaceValidity:
    "Certify changed restriction rows without repeating inherited proofs."
    changed_blocks = tuple(
        CellBlock(
            block.name,
            block.cell_kind,
            np.asarray(block.vertices)[mask],
            global_ids=np.asarray(block.global_ids)[mask],
        )
        for block, mask in zip(target_mesh.blocks, changed_masks, strict=True)
        if np.any(mask)
    )
    changed_by_name = {
        block.name: mask
        for block, mask in zip(target_mesh.blocks, changed_masks, strict=True)
        if np.any(mask)
    }
    changed_validity = None
    if changed_blocks:
        changed_mesh = CellMesh(
            np.asarray(target_mesh.coordinates),
            changed_blocks,
            vertex_global_ids=target_mesh.vertex_global_ids,
            numeric_version=target_mesh.numeric_version,
        )
        element_lookup = dict(
            zip(target_geometry.block_names, target_geometry.elements, strict=True)
        )
        route_lookup = dict(
            zip(target_geometry.block_names, target_geometry.geometry_dofs, strict=True)
        )
        origin = target_geometry.restriction_source
        if origin is None:
            raise RuntimeError(
                "surface_chart.certify_born_surface_cells.pieces must satisfy the material reference contract; got Changed restricted cells lost their source record."
            )
        source_blocks = origin.block_source_blocks
        changed_origin = CellGeometryRestrictionSource(
            origin.source_geometry_id,
            origin.source_topology_id,
            {
                name: np.asarray(origin.block_parent_cell_ids[name])[mask]
                for name, mask in changed_by_name.items()
            },
            {
                name: np.asarray(origin.block_parent_vertex_ids[name])[mask]
                for name, mask in changed_by_name.items()
            },
            block_source_blocks=None
            if source_blocks is None
            else {name: source_blocks[name] for name in changed_by_name},
        )
        changed_geometry = CellGeometrySpec(
            {name: element_lookup[name] for name in changed_by_name},
            {
                name: np.asarray(route_lookup[name])[mask]
                for name, mask in changed_by_name.items()
            },
            target_geometry.coordinates,
            restriction_source=changed_origin,
            budget=budget,
        )
        changed_validity = certify_cell_geometry_validity(
            changed_geometry, mesh=changed_mesh, policy=policy, budget=budget
        )
        if not changed_validity.all_certified:
            raise ValueError(
                "surface_chart.certify_born_surface_cells.validity must satisfy the positive cell validity contract; got Born restricted surface cells are not certified valid."
            )
    return _BornSurfaceValidity(changed_validity=changed_validity)


class _SurfaceValiditySourceRows(StrictModule):
    source_rows: dict[int, tuple[int, str, tuple[int, ...], tuple[int, ...]]]
    source_witness_rows: dict[int, int]
    target_witness_rows: dict[int, int]


def _surface_validity_source_rows(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    source_witness: surface_chart_records.SurfaceChartWitness,
    target_witness: surface_chart_records.SurfaceChartWitness,
    budget: CoordinateEnclosureBudget,
) -> _SurfaceValiditySourceRows:
    "Index retained validity rows by scientific cell and chart identities."
    source_elements, source_routes, _ = source_geometry.resolve(
        source_mesh, budget=budget
    )
    source_rows: dict[int, tuple[int, str, tuple[int, ...], tuple[int, ...]]] = {}
    cursor = 0
    source_vertex_ids = np.asarray(source_mesh.vertex_global_ids)
    for block, element, routes in zip(
        source_mesh.blocks, source_elements, source_routes, strict=True
    ):
        for local_row, (identifier, vertices, route) in enumerate(
            zip(
                np.asarray(block.global_ids),
                np.asarray(block.vertices),
                np.asarray(routes),
                strict=True,
            )
        ):
            source_rows[int(identifier)] = (
                cursor + local_row,
                element.element_id,
                tuple(map(int, route)),
                tuple(map(int, source_vertex_ids[vertices])),
            )
        cursor += block.cell_count
    source_witness_rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(source_witness.cell_global_ids))
    }
    target_witness_rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(target_witness.cell_global_ids))
    }

    return _SurfaceValiditySourceRows(
        source_rows=source_rows,
        source_witness_rows=source_witness_rows,
        target_witness_rows=target_witness_rows,
    )


class _SurfaceValidityInheritance(StrictModule):
    changed_masks: list[NDArray[np.bool_]]
    inherited_rows: list[list[tuple[int, Fraction] | None]]


def _surface_validity_inheritance(
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    source_witness: surface_chart_records.SurfaceChartWitness,
    target_witness: surface_chart_records.SurfaceChartWitness,
    source_rows: dict[int, tuple[int, str, tuple[int, ...], tuple[int, ...]]],
    source_witness_rows: dict[int, int],
    target_witness_rows: dict[int, int],
    budget: CoordinateEnclosureBudget,
) -> _SurfaceValidityInheritance:
    "Prepare changed-row masks and positive-area validity inheritance routes."
    target_elements, target_routes, _ = target_geometry.resolve(
        target_mesh, budget=budget
    )
    target_vertex_ids = np.asarray(target_mesh.vertex_global_ids)
    changed_masks: list[np.ndarray] = []
    inherited_rows: list[list[tuple[int, Fraction] | None]] = []
    source_ids = sorted(source_witness_rows)
    source_patches = np.asarray(source_witness.patches)
    source_charts = np.asarray(source_witness.charts)
    target_patches = np.asarray(target_witness.patches)
    target_charts = np.asarray(target_witness.charts)
    for block, element, routes in zip(
        target_mesh.blocks, target_elements, target_routes, strict=True
    ):
        changed = np.ones((block.cell_count,), dtype=np.bool_)
        inherited: list[tuple[int, Fraction] | None] = []
        for row, (identifier, vertices, route) in enumerate(
            zip(
                np.asarray(block.global_ids),
                np.asarray(block.vertices),
                np.asarray(routes),
                strict=True,
            )
        ):
            identifier_ = int(identifier)
            previous = source_rows.get(identifier_)
            inheritance: tuple[int, Fraction] | None = None
            if previous is not None and previous[1:] == (
                element.element_id,
                tuple(map(int, route)),
                tuple(map(int, target_vertex_ids[vertices])),
            ):
                inheritance = previous[0], Fraction(1)
            else:
                target_row = target_witness_rows[identifier_]
                for source_id in source_ids:
                    source_row = source_witness_rows[source_id]
                    if source_patches[source_row] != target_patches[target_row]:
                        continue
                    scale = _restriction_scale(
                        source_charts[source_row], target_charts[target_row]
                    )
                    if scale is not None:
                        inheritance = source_rows[source_id][0], scale
                        break
            if inheritance is not None:
                changed[row] = False
            inherited.append(inheritance)
        changed_masks.append(changed)
        inherited_rows.append(inherited)
    return _SurfaceValidityInheritance(
        changed_masks=changed_masks, inherited_rows=inherited_rows
    )


def _renew_restricted_surface_validity(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    source_validity: CellValidityCertificate,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    policy: CellValidityPolicy,
    source_witness: surface_chart_records.SurfaceChartWitness,
    target_witness: surface_chart_records.SurfaceChartWitness,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[
    CellValidityCertificate,
    tuple[tuple[int, Fraction] | None, ...],
]:
    "Reuse unchanged restriction rows and certify only born chart cells."
    source_validity.require_bound(source_geometry, mesh=source_mesh)
    if (
        source_validity.policy_id != policy.policy_id
        or not source_validity.all_certified
        or source_geometry.restriction_source is None
        or target_geometry.restriction_source is None
        or source_geometry.restriction_source.source_geometry_id
        != target_geometry.restriction_source.source_geometry_id
        or source_geometry.restriction_source.source_topology_id
        != target_geometry.restriction_source.source_topology_id
        or not np.array_equal(
            np.asarray(source_geometry.coordinates),
            np.asarray(target_geometry.coordinates),
        )
    ):
        raise ValueError(
            "surface_chart.renew_restricted_surface_validity.validity must satisfy the positive cell validity contract; got Restricted validity reuse requires one positive source contract."
        )
    _phase = _surface_validity_source_rows(
        source_mesh=source_mesh,
        source_geometry=source_geometry,
        source_witness=source_witness,
        target_witness=target_witness,
        budget=budget,
    )
    source_rows = _phase.source_rows
    source_witness_rows = _phase.source_witness_rows
    target_witness_rows = _phase.target_witness_rows
    _phase = _surface_validity_inheritance(
        target_mesh=target_mesh,
        target_geometry=target_geometry,
        source_witness=source_witness,
        target_witness=target_witness,
        source_rows=source_rows,
        source_witness_rows=source_witness_rows,
        target_witness_rows=target_witness_rows,
        budget=budget,
    )
    changed_masks = _phase.changed_masks
    inherited_rows = _phase.inherited_rows
    _phase = _certify_born_surface_cells(
        target_mesh=target_mesh,
        target_geometry=target_geometry,
        changed_masks=changed_masks,
        policy=policy,
        budget=budget,
    )
    changed_validity = _phase.changed_validity
    status: list[int] = []
    lower: list[float] = []
    upper: list[float] = []
    depth: list[int] = []
    measure_routes: list[tuple[int, Fraction] | None] = []
    changed_cursor = 0
    target_offsets = [0]
    for block, mask, inherited in zip(
        target_mesh.blocks, changed_masks, inherited_rows, strict=True
    ):
        for is_changed, inheritance in zip(mask, inherited, strict=True):
            scale = Fraction(1)
            if is_changed:
                if changed_validity is None:
                    raise RuntimeError(
                        "surface_chart.renew_restricted_surface_validity.validity must satisfy the positive cell validity contract; got Changed validity rows lost their certificate."
                    )
                row = changed_cursor
                changed_cursor += 1
                certificate = changed_validity
                measure_routes.append(None)
            else:
                if inheritance is None:
                    raise RuntimeError(
                        "surface_chart.renew_restricted_surface_validity.validity must satisfy the positive cell validity contract; got Inherited validity row lost its source proof."
                    )
                row, scale = inheritance
                certificate = source_validity
                measure_routes.append((row, scale))
            status.append(int(np.asarray(certificate.status)[row]))
            low = Fraction(float(np.asarray(certificate.determinant_lower)[row]))
            high = Fraction(float(np.asarray(certificate.determinant_upper)[row]))
            determinant_scale = scale * scale
            lower.append(
                fraction_endpoint(low * determinant_scale, upper=False, saturate=True)
            )
            upper.append(
                fraction_endpoint(high * determinant_scale, upper=True, saturate=True)
            )
            depth.append(int(np.asarray(certificate.depth)[row]))
        target_offsets.append(len(status))
    certificate = CellValidityCertificate(
        np.asarray(status, dtype=np.int32),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        np.asarray(depth, dtype=np.int32),
        block_names=tuple(block.name for block in target_mesh.blocks),
        block_offsets=tuple(target_offsets),
        unsupported_block_names=(),
        unresolved_reasons=(),
        geometry_id=cell_geometry_id(target_geometry),
        geometry_layout_id=target_geometry.geometry_layout_id,
        topology_id=target_mesh.topology_id,
        policy_id=policy.policy_id,
    )
    return certificate, tuple(measure_routes)


def _source_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    witness: surface_chart_records.SurfaceChartWitness,
    order: _IntArray,
    domain: MeshingDomain,
    maximum_work: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[_FloatArray, _FloatArray, _FloatArray, _FloatArray, int]:

    unresolved_elements, routes, _ = geometry.resolve(mesh, budget=budget)
    elements = _coordinate_elements(unresolved_elements)
    local_coordinates = geometry.source_coordinates(budget=budget)
    total, defects, corners, chords = [], [], [], []
    work, offset = 0, 0
    variables = axes(2, budget=budget)
    reference_corners = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        for local in np.asarray(route, dtype=np.int64):
            expressions = coordinate_expressions(
                element,
                tuple(local_coordinates[index] for index in local),
                budget=budget,
            )
            if expressions is None:
                raise ValueError(
                    "surface_chart.source_fidelity.pieces must satisfy the material reference contract; got Source coordinate map lacks ordered source expressions."
                )
            work += (
                sum(
                    expression_node_count(expression, "simplex", 2)
                    for expression in expressions
                )
                + element.local_dof_count
            )
            if work > maximum_work:
                raise surface_chart_records.SurfaceChartResourceError(
                    "surface_chart.source_fidelity.budget must satisfy the admitted resource limits; got Surface source expression enclosure exceeds its work budget.",
                    (("source_expression_work", work, maximum_work),),
                )
            row = int(order[offset])
            patch = int(np.asarray(witness.patches)[row])
            chart = np.asarray(witness.charts)[row : row + 1]
            low, high = imported_cell_geometry_transfer.surface_node_enclosures(
                domain, patch, chart, np.asarray(reference_corners, dtype=np.float64)
            )
            corner_values = tuple(
                tuple(
                    expression_evaluate(expression, point, budget=budget)
                    for expression in expressions
                )
                for point in reference_corners
            )
            defect = Fraction(0)
            for component, expression in enumerate(expressions):
                values = tuple(point[component] for point in corner_values)
                chord = add(
                    constant(values[0], 2, budget=budget),
                    add(
                        scale(variables[0], values[1] - values[0], budget=budget),
                        scale(variables[1], values[2] - values[0], budget=budget),
                        budget=budget,
                    ),
                    budget=budget,
                )
                difference = expression_add(
                    expression, expression_scale(chord, -1, budget=budget), budget=budget
                )
                defect += max(
                    abs(value)
                    for value in expression_bernstein_coefficients(
                        difference, "simplex", 2, budget=budget
                    )
                )
            corner_error = max(
                sum(
                    (
                        max(
                            abs(value - Fraction(float(lower))),
                            abs(value - Fraction(float(upper))),
                        )
                        for value, lower, upper in zip(
                            point, low[0, slot], high[0, slot], strict=True
                        )
                    ),
                    Fraction(0),
                )
                for slot, point in enumerate(corner_values)
            )
            chord_bound = float(domain.interpolation_bounds(patch, chart)[0])
            if not math.isfinite(chord_bound) or chord_bound < 0:
                raise ValueError(
                    "surface_chart.source_fidelity.pieces must satisfy the material reference contract; got source lacks a continuous chord enclosure."
                )
            total.append(
                fraction_endpoint(
                    rational_coefficient(
                        defect + corner_error + Fraction(chord_bound),
                        role="surface_chart.source_fidelity.total",
                    ),
                    upper=True,
                    saturate=True,
                )
            )
            defects.append(
                fraction_endpoint(
                    rational_coefficient(
                        defect, role="surface_chart.source_fidelity.defect"
                    ),
                    upper=True,
                    saturate=True,
                )
            )
            corners.append(
                fraction_endpoint(
                    rational_coefficient(
                        corner_error, role="surface_chart.source_fidelity.corner_error"
                    ),
                    upper=True,
                    saturate=True,
                )
            )
            chords.append(chord_bound)
            offset += 1
    return (
        np.asarray(total, dtype=np.float64),
        np.asarray(defects, dtype=np.float64),
        np.asarray(corners, dtype=np.float64),
        np.asarray(chords, dtype=np.float64),
        work,
    )


class _SurfaceEndpointCertificates(StrictModule):
    source_validity: CellValidityCertificate
    target_validity: CellValidityCertificate
    target_measure_routes: tuple[tuple[int, Fraction] | None, ...] | None
    validities: list[CellValidityCertificate]
    embeddings: list[GlobalEmbeddingCertificate]


def _certify_surface_endpoints(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    source_witness: surface_chart_records.SurfaceChartWitness,
    target_witness: surface_chart_records.SurfaceChartWitness,
    domain: MeshingDomain,
    limits: MeshCertificateLimits,
    validity_policy_: CellValidityPolicy,
    prepared_source_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None,
    budget: CoordinateEnclosureBudget,
) -> _SurfaceEndpointCertificates:
    """Authenticate source proofs and certify independent or restricted targets.

    Only endpoints with restriction metadata can inherit validity and embedding.
    Independently reconstructed maps retain source proofs but certify the target.
    """
    _endpoint = certify_chart_endpoint(
        source_mesh,
        source_geometry,
        validity_policy_,
        limits,
        prepared_source_certificates,
        budget=budget,
        stale_message="surface_chart.certify_surface_endpoints.validity must satisfy the positive cell validity contract; got Prepared source certificates must retain the source validity and embedding policies.",
    )
    source_validity, source_embedding = _endpoint.validity, _endpoint.embedding
    restricted_reuse = prepared_source_certificates is not None and (
        source_geometry.restriction_source is not None
        or target_geometry.restriction_source is not None
    )
    if restricted_reuse:
        target_validity, target_measure_routes = _renew_restricted_surface_validity(
            source_mesh,
            source_geometry,
            source_validity,
            target_mesh,
            target_geometry,
            validity_policy_,
            source_witness,
            target_witness,
            budget=budget,
        )
    else:
        target_validity = certify_cell_geometry_validity(
            target_geometry, mesh=target_mesh, policy=validity_policy_, budget=budget
        )
        target_measure_routes = None
    partition_work = (
        _exact_chart_partition_proof(source_witness, target_witness)
        if prepared_source_certificates is not None
        else 0
    )
    if restricted_reuse:
        relation = target_mesh.topology.incidences[1].relation
        valid = np.asarray(relation.valid, dtype=np.bool_)
        incident = np.bincount(
            np.asarray(relation.source_indices)[valid],
            minlength=target_mesh.entity_set(1).count,
        )
        binding = MeshCertificateBinding(
            target_mesh,
            target_geometry,
            source_embedding.binding.coordinate_scope,
            limits,
            source_id=domain.source_id,
            source_revision=domain.source_revision,
        )
        target_embedding = GlobalEmbeddingCertificate(
            binding,
            target_validity.certificate_id,
            (),
            (
                "restricted_source_embedding",
                "target_chart_topology",
                "target_cell_validity",
            ),
            counts=EmbeddingCounts(
                cell_count=target_mesh.entity_set(2).count,
                boundary_facet_count=int(np.count_nonzero(incident == 1)),
                shell_count=source_embedding.shell_count,
                candidate_pair_count=partition_work,
                ray_test_count=0,
                subdivision_piece_count=0,
            ),
        )
    else:
        target_embedding = certify_global_embedding(
            target_mesh, target_geometry, target_validity, limits=limits, budget=budget
        )
    validities = [source_validity, target_validity]
    embeddings = [source_embedding, target_embedding]
    if any(not validity.all_certified for validity in validities) or any(
        embedding.status != "certified" for embedding in embeddings
    ):
        raise ValueError(
            "surface_chart.certify_surface_endpoints.pieces must satisfy the material reference contract; got Surface deformation requires independently valid globally embedded source and target maps."
        )
    return _SurfaceEndpointCertificates(
        source_validity=source_validity,
        target_validity=target_validity,
        target_measure_routes=target_measure_routes,
        validities=validities,
        embeddings=embeddings,
    )


class _SurfaceMeasureEnclosures(StrictModule):
    source_measures: _FloatArray
    target_measures: _FloatArray
    source_errors: _FloatArray
    target_errors: _FloatArray


def _surface_measure_enclosures(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    source_validity: CellValidityCertificate,
    target_validity: CellValidityCertificate,
    target_measure_routes: tuple[tuple[int, Fraction] | None, ...] | None,
    prepared_source_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None,
    measure_absolute_tolerance: float,
    measure_relative_tolerance: float,
    maximum_measure_work: int,
    maximum_measure_subcells: int,
    maximum_binomial_terms: int,
    budget: CoordinateEnclosureBudget,
) -> _SurfaceMeasureEnclosures:
    "Return positive physical areas and their publication error bounds."
    if prepared_source_certificates is not None:
        source_measures, source_errors = _validity_measure_enclosures(source_validity)
        if target_measure_routes is None or any(
            route is None for route in target_measure_routes
        ):
            target_measures, target_errors = _validity_measure_enclosures(target_validity)
        else:
            target_values = []
            target_measure_bounds = []
            for route in target_measure_routes:
                if route is None:
                    raise RuntimeError(
                        "surface_chart.surface_measure_enclosures.measures must satisfy the positive physical measure contract; got Target measure route unexpectedly vanished."
                    )
                source_row, scale = route
                midpoint = Fraction(float(source_measures[source_row])) * scale
                value = float(midpoint)
                error = Fraction(float(source_errors[source_row])) * scale + abs(
                    Fraction(value) - midpoint
                )
                target_values.append(value)
                target_measure_bounds.append(
                    fraction_endpoint(error, upper=True, saturate=True) if error else 0.0
                )
            target_measures = np.asarray(target_values, dtype=np.float64)
            target_errors = np.asarray(target_measure_bounds, dtype=np.float64)
    else:
        source_measures, source_errors, _ = mapped_measure_owner.certified_cell_measures(
            source_mesh,
            source_geometry,
            absolute_tolerance=measure_absolute_tolerance,
            relative_tolerance=measure_relative_tolerance,
            maximum_work=maximum_measure_work,
            maximum_subcells=maximum_measure_subcells,
            maximum_binomial_terms=maximum_binomial_terms,
            budget=budget,
        )
        target_measures, target_errors, _ = mapped_measure_owner.certified_cell_measures(
            target_mesh,
            target_geometry,
            absolute_tolerance=measure_absolute_tolerance,
            relative_tolerance=measure_relative_tolerance,
            maximum_work=maximum_measure_work,
            maximum_subcells=maximum_measure_subcells,
            maximum_binomial_terms=maximum_binomial_terms,
            budget=budget,
        )
    if (
        any(
            np.any(~np.isfinite(values))
            for values in (source_measures, target_measures, source_errors, target_errors)
        )
        or np.any(source_errors < 0)
        or np.any(target_errors < 0)
        or np.any(source_measures <= source_errors)
        or np.any(target_measures <= target_errors)
    ):
        raise ValueError(
            "surface_chart.surface_measure_enclosures.measures must satisfy the positive physical measure contract; got Physical surface areas must have quantitative strictly positive enclosures."
        )
    return _SurfaceMeasureEnclosures(
        source_measures=source_measures,
        target_measures=target_measures,
        source_errors=source_errors,
        target_errors=target_errors,
    )


class _TargetSurfaceFidelity(StrictModule):
    target_bounds: _FloatArray


def _target_surface_fidelity(
    reconstruction: chart_reconstruction.SurfaceGeometryReconstruction,
    target_mesh: CellMesh,
    target_witness: surface_chart_records.SurfaceChartWitness,
    target_order: _IntArray,
    source_bounds: _FloatArray,
    maximum_fidelity: float,
) -> _TargetSurfaceFidelity:
    "Bind target reconstruction rows and enforce the fidelity limit."
    reconstruction_ids = np.asarray(reconstruction.cell_global_ids)
    if np.unique(
        reconstruction_ids
    ).size != reconstruction_ids.size or not np.array_equal(
        np.sort(reconstruction_ids),
        np.sort(surface_chart_records.chart_cell_ids(target_mesh)),
    ):
        raise ValueError(
            "surface_chart.target_surface_fidelity.pieces must satisfy the material reference contract; got Target reconstruction cell IDs do not cover the physical target."
        )
    lookup = {int(identifier): row for row, identifier in enumerate(reconstruction_ids)}
    target_bound_rows = np.asarray(
        [
            lookup[int(identifier)]
            for identifier in surface_chart_records.chart_cell_ids(target_mesh)
        ],
        dtype=np.int64,
    )
    target_bounds = np.asarray(reconstruction.fidelity_bounds, dtype=np.float64)[
        target_bound_rows
    ]
    if (
        np.any(~np.isfinite(target_bounds))
        or np.any(target_bounds < 0)
        or max(float(np.max(source_bounds)), float(np.max(target_bounds)))
        > maximum_fidelity
    ):
        raise ValueError(
            "surface_chart.target_surface_fidelity.fidelity must satisfy the declared fidelity bound; got Continuous source or target fidelity exceeds the accepted bound."
        )
    for physical_row, witness_row in enumerate(target_order):
        row = lookup[int(surface_chart_records.chart_cell_ids(target_mesh)[physical_row])]
        if (
            reconstruction.cell_geometry_entity_ids[row]
            != target_witness.geometry_entity_ids[witness_row]
            or reconstruction.cell_occurrence_paths[row]
            != target_witness.occurrence_paths[witness_row]
        ):
            raise ValueError(
                "surface_chart.target_surface_fidelity.binding must satisfy the declared geometry and cell identities; got Target reconstruction and binding occurrence identities disagree."
            )
        if int(np.asarray(reconstruction.cell_patches)[row]) != int(
            np.asarray(target_witness.patches)[witness_row]
        ) or not np.array_equal(
            np.asarray(reconstruction.cell_charts)[row],
            np.asarray(target_witness.charts)[witness_row],
        ):
            raise ValueError(
                "surface_chart.target_surface_fidelity.binding must satisfy the declared geometry and cell identities; got Target binding changes the reconstruction's reference-to-chart mapping."
            )
    return _TargetSurfaceFidelity(target_bounds=target_bounds)


class _SurfaceChartEndpoints(StrictModule):
    evaluations: int
    candidates: int
    memory: int
    target_geometry: CellGeometrySpec
    source_order: _IntArray
    target_order: _IntArray
    source_patches: set[int]


def _bind_surface_chart_endpoints(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    reconstruction: chart_reconstruction.SurfaceGeometryReconstruction,
    domain: MeshingDomain,
    source_witness: surface_chart_records.SurfaceChartWitness,
    target_witness: surface_chart_records.SurfaceChartWitness,
    maximum_evaluations: int,
    maximum_candidate_pairs: int,
    maximum_memory_bytes: int,
    maximum_fidelity: float,
    maximum_displacement: float,
    prepared_source_fidelity: SourceFidelityCertificate | None,
    prepared_source_restriction_id: str | None,
) -> _SurfaceChartEndpoints:
    "Validate endpoint identities and normalize chart preparation limits."
    evaluations = positive_integer(maximum_evaluations, "maximum_evaluations")
    candidates = positive_integer(maximum_candidate_pairs, "maximum_candidate_pairs")
    memory = positive_integer(maximum_memory_bytes, "maximum_memory_bytes")
    for value in (maximum_fidelity, maximum_displacement):
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "surface_chart.bind_surface_chart_endpoints.fidelity must satisfy the declared fidelity bound; got Continuous fidelity and displacement limits must be finite and nonnegative."
            )
    target_geometry = reconstruction.geometry
    for mesh in (source_mesh, target_mesh):
        if (
            mesh.topological_dimension != 2
            or mesh.ambient_dimension != 3
            or any((block.cell_kind != "triangle" for block in mesh.blocks))
        ):
            raise ValueError(
                "surface_chart.bind_surface_chart_endpoints.pieces must satisfy the material reference contract; got Surface deformation requires explicit physical triangle meshes in 3D."
            )
        mesh.require_dense("Surface chart deformation")
        # Physical quotient copies retain their explicit occurrence UV uses;
        # the periodic controller separately certifies the quotient.
    if (
        reconstruction.source_geometry_id,
        reconstruction.source_topology_id,
        reconstruction.target_topology_id,
        reconstruction.domain_id,
    ) != (
        cell_geometry_id(source_geometry),
        source_mesh.topology_id,
        target_mesh.topology_id,
        domain.domain_id,
    ):
        raise ValueError(
            "surface_chart.bind_surface_chart_endpoints.binding must satisfy the declared geometry and cell identities; got Target reconstruction is stale for this source, topology or domain."
        )
    source_order = _witness_order(source_witness, source_mesh, source_geometry, domain)
    if (prepared_source_fidelity is None) != (prepared_source_restriction_id is None):
        raise ValueError(
            "surface_chart.bind_surface_chart_endpoints.fidelity must satisfy the declared fidelity bound; got Prepared source fidelity and restriction identity must be supplied together."
        )
    if prepared_source_fidelity is not None:
        prepared_source_fidelity.binding.require(source_mesh, source_geometry)
        restriction = source_geometry.restriction_source
        if (
            restriction is None
            or restriction.restriction_source_id != prepared_source_restriction_id
        ):
            raise ValueError(
                "surface_chart.bind_surface_chart_endpoints.fidelity must satisfy the declared fidelity bound; got Prepared source fidelity names another restriction."
            )
        binding = prepared_source_fidelity.binding
        if (
            (binding.source_id, binding.source_revision)
            != (domain.source_id, domain.source_revision)
            or source_witness.geometry_id != binding.geometry_id
            or source_witness.topology_id != binding.topology_id
            or source_witness.domain_id != domain.domain_id
            or prepared_source_fidelity.status != "certified"
            or prepared_source_fidelity.mesh_to_source_semantics != "certified"
            or prepared_source_fidelity.source_to_mesh_semantics != "certified"
            or any(
                value != 0.0
                for value in (
                    prepared_source_fidelity.mesh_to_source_upper,
                    prepared_source_fidelity.mesh_to_source_lower,
                    prepared_source_fidelity.source_to_mesh_upper,
                    prepared_source_fidelity.source_to_mesh_lower,
                )
            )
        ):
            raise ValueError(
                "surface_chart.bind_surface_chart_endpoints.fidelity must satisfy the declared fidelity bound; got Prepared source fidelity is stale, nonzero, or differently governed."
            )
    target_order = _witness_order(target_witness, target_mesh, target_geometry, domain)
    source_patches, target_patches = (
        set(map(int, np.asarray(source_witness.patches))),
        set(map(int, np.asarray(target_witness.patches))),
    )
    if source_patches != target_patches:
        raise ValueError(
            "surface_chart.bind_surface_chart_endpoints.pieces must satisfy the material reference contract; got Source and target must cover the same separate patch occurrences."
        )
    return _SurfaceChartEndpoints(
        evaluations=evaluations,
        candidates=candidates,
        memory=memory,
        target_geometry=target_geometry,
        source_order=source_order,
        target_order=target_order,
        source_patches=source_patches,
    )


class _SurfaceMaterialOccurrences(StrictModule):
    occurrences: list[surface_chart_records.PreparedSurfaceChartOccurrence]
    work: int
    maximum_bound: float


class _SurfaceOccurrenceCover(StrictModule):
    remaining_candidates: int


def _certify_surface_occurrence_cover(
    domain: MeshingDomain,
    patch: int,
    root_carrier: CellMesh,
    target_carrier: CellMesh,
    source_chart_cover: MeshingDomainBoundarySource | None,
    target_chart_cover: MeshingDomainBoundarySource | None,
    prepared_source_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None,
    remaining_candidates: int,
    memory: int,
    retained: int,
    budget: CoordinateEnclosureBudget,
    native_ticket: NativeBatchTicket,
) -> _SurfaceOccurrenceCover:
    "Certify root and target trim coverage under the shared candidate limit."
    if prepared_source_certificates is None:
        if source_chart_cover is None:
            _domain_cover(
                domain,
                patch,
                root_carrier,
                budget=budget,
                native_ticket=native_ticket,
            )
        else:
            remaining_candidates -= _source_owned_domain_cover(
                domain,
                patch,
                root_carrier,
                source_chart_cover,
                remaining_candidates,
                memory - retained,
                budget=budget,
                native_ticket=native_ticket,
            )
        target_cover = (
            source_chart_cover if target_chart_cover is None else target_chart_cover
        )
        if target_cover is None:
            _domain_cover(
                domain,
                patch,
                target_carrier,
                budget=budget,
                native_ticket=native_ticket,
            )
        else:
            remaining_candidates -= _source_owned_domain_cover(
                domain,
                patch,
                target_carrier,
                target_cover,
                remaining_candidates,
                memory - retained,
                budget=budget,
                native_ticket=native_ticket,
            )
    # The root condition is independently transported to every current source
    # cell by a whole native partition, not a UV proxy.
    return _SurfaceOccurrenceCover(remaining_candidates=remaining_candidates)


def _surface_material_occurrences(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    domain: MeshingDomain,
    source_witness: surface_chart_records.SurfaceChartWitness,
    target_witness: surface_chart_records.SurfaceChartWitness,
    source_order: _IntArray,
    target_order: _IntArray,
    source_patches: set[int],
    root_witness: surface_chart_records.SurfaceChartWitness,
    source_material: tuple[_ExactTriangle, ...],
    target_material: tuple[_ExactTriangle, ...],
    root_material: tuple[_ExactTriangle, ...],
    source_chart_cover: MeshingDomainBoundarySource | None,
    target_chart_cover: MeshingDomainBoundarySource | None,
    prepared_source_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None,
    prepared_source_fidelity: SourceFidelityCertificate | None,
    remaining_candidates: int,
    evaluations: int,
    memory: int,
    retained: int,
    work: int,
    maximum_displacement: float,
    budget: CoordinateEnclosureBudget,
    native_ticket: NativeBatchTicket,
) -> _SurfaceMaterialOccurrences:
    "Construct ordered occurrence overlaps within remaining work and storage."
    occurrences: list[surface_chart_records.PreparedSurfaceChartOccurrence] = []
    source_rows_by_witness = np.empty(source_order.size, dtype=np.int64)
    source_rows_by_witness[source_order] = np.arange(source_order.size, dtype=np.int64)
    maximum_bound = 0.0
    for patch in sorted(
        source_patches,
        key=lambda index: (
            domain.entity_id(2, index),
            domain.source_occurrences[2][index],
        ),
    ):
        target_carrier, target_rows, _ = _carrier(
            target_witness, target_order, patch, budget=budget
        )
        root_carrier, _, _ = _carrier(
            root_witness,
            np.arange(root_witness.cell_global_ids.shape[0], dtype=np.int64),
            patch,
            budget=budget,
        )
        old_selection = np.flatnonzero(np.asarray(source_witness.patches) == patch)
        source_rows = source_rows_by_witness[old_selection]
        new_selection = np.flatnonzero(np.asarray(target_witness.patches) == patch)
        root_selection = np.flatnonzero(np.asarray(root_witness.patches) == patch)
        old_charts = tuple(source_material[int(row)] for row in old_selection)
        new_charts = tuple(target_material[int(row)] for row in new_selection)
        root_charts = tuple(root_material[int(row)] for row in root_selection)
        old_ids = np.asarray(source_witness.cell_global_ids)[old_selection]
        new_ids = np.asarray(target_witness.cell_global_ids)[new_selection]
        root_ids = np.asarray(root_witness.cell_global_ids)[root_selection]
        if not root_charts:
            raise ValueError(
                "surface_chart.surface_material_occurrences.binding must satisfy the declared geometry and cell identities; got Material root binding omits an source occurrence."
            )
        _phase = _certify_surface_occurrence_cover(
            domain=domain,
            patch=patch,
            root_carrier=root_carrier,
            target_carrier=target_carrier,
            source_chart_cover=source_chart_cover,
            target_chart_cover=target_chart_cover,
            prepared_source_certificates=prepared_source_certificates,
            remaining_candidates=remaining_candidates,
            memory=memory,
            retained=retained,
            budget=budget,
            native_ticket=native_ticket,
        )
        remaining_candidates = _phase.remaining_candidates
        root_pieces = ()
        if old_charts != root_charts or not np.array_equal(old_ids, root_ids):
            root_pieces, root_work, root_pairs, root_bytes = (
                prepare_exact_material_pieces(
                    root_charts,
                    old_charts,
                    root_ids,
                    old_ids,
                    remaining_candidates,
                    evaluations - work,
                    memory - retained,
                    partitions_certified=prepared_source_certificates is not None,
                    budget=budget,
                )
            )
            remaining_candidates -= root_pairs
            work += root_work
            retained += root_bytes
        pieces, overlap_work, pair_count, piece_bytes = prepare_exact_material_pieces(
            old_charts,
            new_charts,
            old_ids,
            new_ids,
            remaining_candidates,
            evaluations - work,
            memory - retained,
            partitions_certified=prepared_source_certificates is not None,
            budget=budget,
        )
        remaining_candidates -= pair_count
        work += overlap_work
        retained += piece_bytes
        if prepared_source_fidelity is not None:
            displacement = np.zeros((len(pieces),), dtype=np.float64)
            difference_work = 0
        else:
            displacement, difference_work = _material_displacement_bounds(
                source_mesh,
                source_geometry,
                target_mesh,
                target_geometry,
                source_rows,
                target_rows,
                pieces,
                evaluations - work,
                memory - retained,
                budget=budget,
            )
        work += difference_work
        maximum_bound = max(maximum_bound, float(np.max(displacement, initial=0.0)))
        if maximum_bound > maximum_displacement:
            raise ValueError(
                "surface_chart.surface_material_occurrences.displacement must satisfy the whole-piece displacement bound; got Old-to-target whole-material-overlap displacement exceeds the accepted bound."
            )
        occurrences.append(
            surface_chart_records.PreparedSurfaceChartOccurrence(
                patch=patch,
                geometry_entity_id=domain.entity_id(2, patch),
                occurrence_path=domain.source_occurrences[2][patch],
                source_rows=jnp.asarray(source_rows, dtype=jnp.int64),
                target_rows=jnp.asarray(target_rows, dtype=jnp.int64),
                exact_source_material_charts=old_charts,
                exact_target_material_charts=new_charts,
                pieces=pieces,
                root_pieces=root_pieces,
                material_root_witness_id=root_witness.witness_id,
                displacement_bounds=jnp.asarray(displacement, dtype=jnp.float64),
                source_domain_coverage="certified",
                target_domain_coverage="certified",
            )
        )
    return _SurfaceMaterialOccurrences(
        occurrences=occurrences, work=work, maximum_bound=maximum_bound
    )


class _SurfaceSourceFidelity(StrictModule):
    cell_count: int
    source_bounds: _FloatArray
    polynomial: _FloatArray
    corner: _FloatArray
    chord: _FloatArray
    work: int


def _prepare_surface_source_fidelity(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    source_witness: surface_chart_records.SurfaceChartWitness,
    target_witness: surface_chart_records.SurfaceChartWitness,
    domain: MeshingDomain,
    source_order: _IntArray,
    target_order: _IntArray,
    source_patches: set[int],
    source_chart_cover: MeshingDomainBoundarySource | None,
    prepared_source_fidelity: SourceFidelityCertificate | None,
    evaluations: int,
    memory: int,
    budget: CoordinateEnclosureBudget,
) -> _SurfaceSourceFidelity:
    "Admit preparation costs before enclosing the source coordinate map."
    cell_count = source_order.size + target_order.size
    # Include carrier connectivity, coordinates, maps and retained certificate arrays.
    unresolved_source_elements, source_routes, _ = source_geometry.resolve(
        source_mesh, budget=budget
    )
    source_elements = _coordinate_elements(unresolved_source_elements)
    polynomial_work = sum(
        np.asarray(route).shape[0]
        * element.local_dof_count
        * 3
        * math.comb(element.degree + 2, 2)
        for element, route in zip(source_elements, source_routes, strict=True)
    )
    if source_chart_cover is None:
        boundary_work = sum(
            9 * int(np.sum(np.asarray(witness.patches) == patch)) ** 2
            for witness in (source_witness, target_witness)
            for patch in source_patches
        )
    else:
        # The retained source contract verifies its boundary once; whole native
        # overlap transfers that proof, rather than re-proving every cell pair.
        boundary_work = sum(
            record[4].shape[0] * (record[4].shape[0] - 1) // 2
            for record in source_chart_cover.chart_triangulations
            if record[0] in source_patches
        )
    preparation_work = cell_count * 24 + polynomial_work + boundary_work
    if cell_count * 1024 > memory or preparation_work > evaluations:
        raise surface_chart_records.SurfaceChartResourceError(
            "surface_chart.prepare_surface_source_fidelity.budget must satisfy the admitted resource limits; got Surface chart preparation exceeds resources before artifacts.",
            (
                ("chart_preparation_work_bound", preparation_work, evaluations),
                ("chart_preparation_storage_bound", cell_count * 1024, memory),
            ),
        )
    if prepared_source_fidelity is None:
        source_bounds, polynomial, corner, chord, work = _source_fidelity(
            source_mesh,
            source_geometry,
            source_witness,
            source_order,
            domain,
            evaluations,
            budget=budget,
        )
    else:
        source_bounds = np.zeros((source_order.size,), dtype=np.float64)
        polynomial = source_bounds.copy()
        corner = source_bounds.copy()
        chord = source_bounds.copy()
        work = 0
    return _SurfaceSourceFidelity(
        cell_count=cell_count,
        source_bounds=source_bounds,
        polynomial=polynomial,
        corner=corner,
        chord=chord,
        work=work,
    )


class _SurfaceMaterialBanks(StrictModule):
    root_witness: surface_chart_records.SurfaceChartWitness
    source_material: tuple[_ExactTriangle, ...]
    target_material: tuple[_ExactTriangle, ...]
    root_material: tuple[_ExactTriangle, ...]
    retained: int
    remaining_candidates: int


def _prepare_surface_material_banks(
    source_witness: surface_chart_records.SurfaceChartWitness,
    target_witness: surface_chart_records.SurfaceChartWitness,
    material_root_witness: surface_chart_records.SurfaceChartWitness | None,
    source_material_charts: ConvertibleToArray | None,
    target_material_charts: ConvertibleToArray | None,
    domain: MeshingDomain,
    source_order: _IntArray,
    target_order: _IntArray,
    cell_count: int,
    memory: int,
    candidates: int,
) -> _SurfaceMaterialBanks:
    "Bind endpoint material banks to their retained root chart."
    if (source_material_charts is None) != (target_material_charts is None):
        raise ValueError(
            "surface_chart.prepare_surface_material_banks.binding must satisfy the declared geometry and cell identities; got Material correspondence requires both endpoint chart banks."
        )
    root_witness = (
        source_witness if material_root_witness is None else material_root_witness
    )
    if root_witness.domain_id != domain.domain_id:
        raise ValueError(
            "surface_chart.prepare_surface_material_banks.binding must satisfy the declared geometry and cell identities; got Material root binding changes its source domain."
        )
    if source_material_charts is not None and material_root_witness is None:
        raise ValueError(
            "surface_chart.prepare_surface_material_banks.binding must satisfy the declared geometry and cell identities; got Authored material correspondence requires its retained source root chart binding."
        )
    source_material = surface_chart_records.exact_chart_bank(
        source_witness.charts
        if source_material_charts is None
        else source_material_charts,
        source_order.size,
    )
    target_material = surface_chart_records.exact_chart_bank(
        target_witness.charts
        if target_material_charts is None
        else target_material_charts,
        target_order.size,
    )
    root_material = surface_chart_records.exact_chart_bank(
        root_witness.charts, root_witness.cell_global_ids.shape[0]
    )
    material_bytes = _exact_retained_bytes(
        (source_material, target_material, root_material)
    )
    retained, remaining_candidates = (cell_count * 1024 + material_bytes, candidates)
    if retained >= memory:
        raise surface_chart_records.SurfaceChartResourceError(
            "surface_chart.prepare_surface_material_banks.budget must satisfy the admitted resource limits; got scientific material banks exhaust the source retained allowance.",
            (("exact_material_retained_bytes", retained, memory),),
        )
    return _SurfaceMaterialBanks(
        root_witness=root_witness,
        source_material=source_material,
        target_material=target_material,
        root_material=root_material,
        retained=retained,
        remaining_candidates=remaining_candidates,
    )


# This host lifecycle keeps shared resource accounting and certificate/publication order across the already-decomposed scientific phases.
def _prepare_surface_chart_deformation(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    reconstruction: chart_reconstruction.SurfaceGeometryReconstruction,
    domain: MeshingDomain,
    /,
    *,
    source_witness: surface_chart_records.SurfaceChartWitness,
    target_witness: surface_chart_records.SurfaceChartWitness,
    maximum_fidelity: float,
    maximum_displacement: float,
    maximum_evaluations: int = 1_000_000,
    maximum_candidate_pairs: int = 1_000_000,
    maximum_memory_bytes: int = 256 * 1024**2,
    measure_absolute_tolerance: float = 1e-10,
    measure_relative_tolerance: float = 1e-10,
    maximum_measure_work: int = 100_000_000,
    maximum_measure_subcells: int = 10_000,
    maximum_binomial_terms: int = 32,
    certificate_limits: MeshCertificateLimits | None = None,
    validity_policy: CellValidityPolicy | None = None,
    prepared_source_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None = None,
    source_chart_cover: MeshingDomainBoundarySource | None = None,
    source_material_charts: ConvertibleToArray | None = None,
    target_material_charts: ConvertibleToArray | None = None,
    material_root_witness: surface_chart_records.SurfaceChartWitness | None = None,
    target_chart_cover: MeshingDomainBoundarySource | None = None,
    prepared_source_fidelity: SourceFidelityCertificate | None = None,
    prepared_source_restriction_id: str | None = None,
    budget: CoordinateEnclosureBudget,
    native_ticket: NativeBatchTicket,
) -> surface_chart_records.PreparedSurfaceChartDeformation:
    "Prepare the full bounded transition; any unavailable condition refuses it."

    _phase = _bind_surface_chart_endpoints(
        source_mesh=source_mesh,
        source_geometry=source_geometry,
        target_mesh=target_mesh,
        reconstruction=reconstruction,
        domain=domain,
        source_witness=source_witness,
        target_witness=target_witness,
        maximum_evaluations=maximum_evaluations,
        maximum_candidate_pairs=maximum_candidate_pairs,
        maximum_memory_bytes=maximum_memory_bytes,
        maximum_fidelity=maximum_fidelity,
        maximum_displacement=maximum_displacement,
        prepared_source_fidelity=prepared_source_fidelity,
        prepared_source_restriction_id=prepared_source_restriction_id,
    )
    evaluations = _phase.evaluations
    candidates = _phase.candidates
    memory = _phase.memory
    target_geometry = _phase.target_geometry
    source_order = _phase.source_order
    target_order = _phase.target_order
    source_patches = _phase.source_patches
    _phase = _prepare_surface_source_fidelity(
        source_mesh=source_mesh,
        source_geometry=source_geometry,
        source_witness=source_witness,
        target_witness=target_witness,
        domain=domain,
        source_order=source_order,
        target_order=target_order,
        source_patches=source_patches,
        source_chart_cover=source_chart_cover,
        prepared_source_fidelity=prepared_source_fidelity,
        evaluations=evaluations,
        memory=memory,
        budget=budget,
    )
    cell_count = _phase.cell_count
    source_bounds = _phase.source_bounds
    polynomial = _phase.polynomial
    corner = _phase.corner
    chord = _phase.chord
    work = _phase.work
    _phase = _target_surface_fidelity(
        reconstruction=reconstruction,
        target_mesh=target_mesh,
        target_witness=target_witness,
        target_order=target_order,
        source_bounds=source_bounds,
        maximum_fidelity=maximum_fidelity,
    )
    target_bounds = _phase.target_bounds
    limits = (
        MeshCertificateLimits(
            maximum_candidate_pairs=candidates,
            maximum_source_samples=evaluations,
            maximum_distance_evaluations=evaluations,
            maximum_subdivision_pieces=min(evaluations, memory // 1024),
            maximum_bernstein_nodes=min(evaluations, memory // 1024),
        )
        if certificate_limits is None
        else certificate_limits
    )
    validity_policy_ = (
        CellValidityPolicy(
            maximum_piece_count=min(evaluations, memory // 1024),
            maximum_bernstein_nodes=min(evaluations, memory // 1024),
        )
        if validity_policy is None
        else validity_policy
    )
    _phase = _certify_surface_endpoints(
        source_mesh=source_mesh,
        source_geometry=source_geometry,
        target_mesh=target_mesh,
        target_geometry=target_geometry,
        source_witness=source_witness,
        target_witness=target_witness,
        domain=domain,
        limits=limits,
        validity_policy_=validity_policy_,
        prepared_source_certificates=prepared_source_certificates,
        budget=budget,
    )
    source_validity = _phase.source_validity
    target_validity = _phase.target_validity
    target_measure_routes = _phase.target_measure_routes
    validities = _phase.validities
    embeddings = _phase.embeddings
    _phase = _prepare_surface_material_banks(
        source_witness=source_witness,
        target_witness=target_witness,
        material_root_witness=material_root_witness,
        source_material_charts=source_material_charts,
        target_material_charts=target_material_charts,
        domain=domain,
        source_order=source_order,
        target_order=target_order,
        cell_count=cell_count,
        memory=memory,
        candidates=candidates,
    )
    root_witness = _phase.root_witness
    source_material = _phase.source_material
    target_material = _phase.target_material
    root_material = _phase.root_material
    retained = _phase.retained
    remaining_candidates = _phase.remaining_candidates
    _phase = _surface_material_occurrences(
        source_mesh=source_mesh,
        source_geometry=source_geometry,
        target_mesh=target_mesh,
        target_geometry=target_geometry,
        domain=domain,
        source_witness=source_witness,
        target_witness=target_witness,
        source_order=source_order,
        target_order=target_order,
        source_patches=source_patches,
        root_witness=root_witness,
        source_material=source_material,
        target_material=target_material,
        root_material=root_material,
        source_chart_cover=source_chart_cover,
        target_chart_cover=target_chart_cover,
        prepared_source_certificates=prepared_source_certificates,
        prepared_source_fidelity=prepared_source_fidelity,
        remaining_candidates=remaining_candidates,
        evaluations=evaluations,
        memory=memory,
        retained=retained,
        work=work,
        maximum_displacement=maximum_displacement,
        budget=budget,
        native_ticket=native_ticket,
    )
    occurrences = _phase.occurrences
    work = _phase.work
    maximum_bound = _phase.maximum_bound
    _phase = _surface_measure_enclosures(
        source_mesh=source_mesh,
        source_geometry=source_geometry,
        target_mesh=target_mesh,
        target_geometry=target_geometry,
        source_validity=source_validity,
        target_validity=target_validity,
        target_measure_routes=target_measure_routes,
        prepared_source_certificates=prepared_source_certificates,
        measure_absolute_tolerance=measure_absolute_tolerance,
        measure_relative_tolerance=measure_relative_tolerance,
        maximum_measure_work=maximum_measure_work,
        maximum_measure_subcells=maximum_measure_subcells,
        maximum_binomial_terms=maximum_binomial_terms,
        budget=budget,
    )
    source_measures = _phase.source_measures
    target_measures = _phase.target_measures
    source_errors = _phase.source_errors
    target_errors = _phase.target_errors
    cover_ids = tuple(
        None
        if cover is None
        else canonical_fingerprint(
            {
                "domain": cover.domain.domain_id,
                "patches": cover.patches,
                "resolution": cover.resolution,
                "records": array_tree_fingerprint(cover.chart_triangulations),
            }
        )
        for cover in (
            source_chart_cover,
            source_chart_cover if target_chart_cover is None else target_chart_cover,
        )
    )
    identity = canonical_fingerprint(
        {
            "kind": "prepared-surface-chart-deformation",
            "source": source_witness.witness_id,
            "target": target_witness.witness_id,
            "material_root": root_witness.witness_id,
            "chart_covers": cover_ids,
            "reconstruction": reconstruction.reconstruction_id,
            "prepared_source_fidelity": (
                None
                if prepared_source_fidelity is None
                else (
                    prepared_source_fidelity.certificate_id,
                    prepared_source_restriction_id,
                )
            ),
            "root_partitions": tuple(
                tuple(piece.piece_id for piece in item.root_pieces)
                for item in occurrences
            ),
            "overlaps": tuple(
                tuple(piece.piece_id for piece in item.pieces) for item in occurrences
            ),
            "source_bounds": array_tree_fingerprint(source_bounds),
            "target_bounds": array_tree_fingerprint(target_bounds),
            "source_measures": array_tree_fingerprint(source_measures),
            "target_measures": array_tree_fingerprint(target_measures),
            "source_errors": array_tree_fingerprint(source_errors),
            "target_errors": array_tree_fingerprint(target_errors),
            "embedding": tuple(item.certificate_id for item in embeddings),
        }
    )
    return surface_chart_records.PreparedSurfaceChartDeformation(
        source_witness=source_witness,
        target_witness=target_witness,
        material_root_witness=root_witness,
        material_root_chart_cover=source_chart_cover,
        target_chart_cover=source_chart_cover
        if target_chart_cover is None
        else target_chart_cover,
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
        occurrences=tuple(occurrences),
        source_geometry_id=source_witness.geometry_id,
        target_geometry_id=target_witness.geometry_id,
        source_topology_id=source_mesh.topology_id,
        target_topology_id=target_mesh.topology_id,
        domain_id=domain.domain_id,
        source_fidelity_bounds=jnp.asarray(source_bounds, dtype=jnp.float64),
        target_fidelity_bounds=jnp.asarray(target_bounds, dtype=jnp.float64),
        source_polynomial_chord_bounds=jnp.asarray(polynomial, dtype=jnp.float64),
        source_corner_enclosure_bounds=jnp.asarray(corner, dtype=jnp.float64),
        source_continuous_chord_bounds=jnp.asarray(chord, dtype=jnp.float64),
        source_cell_measures=jnp.asarray(source_measures, dtype=jnp.float64),
        target_cell_measures=jnp.asarray(target_measures, dtype=jnp.float64),
        source_measure_errors=jnp.asarray(source_errors, dtype=jnp.float64),
        target_measure_errors=jnp.asarray(target_errors, dtype=jnp.float64),
        source_validity=validities[0],
        target_validity=validities[1],
        source_embedding=embeddings[0],
        target_embedding=embeddings[1],
        maximum_displacement_bound=maximum_bound,
        evaluation_count=work,
        deformation_id=identity,
    )


@checked
def prepare_surface_chart_deformation(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    reconstruction: chart_reconstruction.SurfaceGeometryReconstruction,
    domain: MeshingDomain,
    /,
    *,
    source_witness: surface_chart_records.SurfaceChartWitness,
    target_witness: surface_chart_records.SurfaceChartWitness,
    maximum_fidelity: float,
    maximum_displacement: float,
    maximum_evaluations: int = 1_000_000,
    maximum_candidate_pairs: int = 1_000_000,
    maximum_memory_bytes: int = 256 * 1024**2,
    measure_absolute_tolerance: float = 1e-10,
    measure_relative_tolerance: float = 1e-10,
    maximum_measure_work: int = 100_000_000,
    maximum_measure_subcells: int = 10_000,
    maximum_binomial_terms: int = 32,
    certificate_limits: MeshCertificateLimits | None = None,
    validity_policy: CellValidityPolicy | None = None,
    prepared_source_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None = None,
    source_chart_cover: MeshingDomainBoundarySource | None = None,
    source_material_charts: ConvertibleToArray | None = None,
    target_material_charts: ConvertibleToArray | None = None,
    material_root_witness: surface_chart_records.SurfaceChartWitness | None = None,
    target_chart_cover: MeshingDomainBoundarySource | None = None,
    prepared_source_fidelity: SourceFidelityCertificate | None = None,
    prepared_source_restriction_id: str | None = None,
    budget: CoordinateEnclosureBudget | None = None,
    scope: NativeExecutionScope | None = None,
) -> surface_chart_records.PreparedSurfaceChartDeformation:
    """Prepare bounded chart correspondence with an explicit request ledger."""
    work = positive_integer(maximum_evaluations, "maximum_evaluations")
    memory = positive_integer(maximum_memory_bytes, "maximum_memory_bytes")
    with chart_execution.chart_native_execution(
        budget,
        scope,
        maximum_work=work,
        maximum_memory=memory,
        maximum_queries=work,
    ) as (budget, native_ticket):
        return _prepare_surface_chart_deformation(
            source_mesh,
            source_geometry,
            target_mesh,
            reconstruction,
            domain,
            source_witness=source_witness,
            target_witness=target_witness,
            maximum_fidelity=maximum_fidelity,
            maximum_displacement=maximum_displacement,
            maximum_evaluations=maximum_evaluations,
            maximum_candidate_pairs=maximum_candidate_pairs,
            maximum_memory_bytes=maximum_memory_bytes,
            measure_absolute_tolerance=measure_absolute_tolerance,
            measure_relative_tolerance=measure_relative_tolerance,
            maximum_measure_work=maximum_measure_work,
            maximum_measure_subcells=maximum_measure_subcells,
            maximum_binomial_terms=maximum_binomial_terms,
            certificate_limits=certificate_limits,
            validity_policy=validity_policy,
            prepared_source_certificates=prepared_source_certificates,
            source_chart_cover=source_chart_cover,
            source_material_charts=source_material_charts,
            target_material_charts=target_material_charts,
            material_root_witness=material_root_witness,
            target_chart_cover=target_chart_cover,
            prepared_source_fidelity=prepared_source_fidelity,
            prepared_source_restriction_id=prepared_source_restriction_id,
            budget=budget,
            native_ticket=native_ticket,
        )


__all__ = ["prepare_surface_chart_deformation"]
