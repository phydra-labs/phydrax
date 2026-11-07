#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact containment, disjointness, and completeness of mapped child charts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import canonical_fingerprint
from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_geometry import (
    CellGeometryElement,
    CellGeometrySpec,
    coordinate_reference_chain,
    restriction_chart_arguments,
    ScalarCoordinateElement,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import (
    affine_map,
    compose_expression,
    ExactScalar,
    Expression,
    expression_equal,
    ExpressionInput,
    RationalPolynomial,
)
from ..discretization._mapped_power_geometry import MappedPowerCellGeometrySource
from ..discretization._nested_reference_types import validate_reference_cell_map_witnesses
from ..discretization._reference_cell import reference_cell_topology
from ..linalg._exact import determinant_exact
from ._domain_numerics import (
    affine_cell_measure,
    cell_coordinate_expressions,
    reference_cell_measure,
)
from ._domain_planar_boundary import oriented_target_facets
from ._embedding_state import (
    MeshCertificateEntityKind,
    MeshCertificateFinding,
    MeshFindingStatus,
)
from ._exact_polyhedral_geometry import tetrahedron_contact_points
from ._mapped_reference_domain import MappedReferenceDomain
from ._mesh_certificates import _same_tree_content
from ._planar_coverage import FacetOverlap, partition_overlaps
from ._planar_primitives import _float_down, _float_up, area2, intersection


if TYPE_CHECKING:
    from ._global_embedding import ValidatedGlobalEmbedding


@dataclass(frozen=True, slots=True)
class MappedChartProof:
    """Exact child-chart ancestry and the independent root-boundary witness."""

    parent_cell_ids: tuple[int, ...]
    chart_ids: tuple[str, ...]
    facet_overlaps: tuple[FacetOverlap, ...]
    findings: tuple[MeshCertificateFinding, ...]
    target_facet_count: int


@dataclass(frozen=True, slots=True)
class _RootCell:
    block_index: int
    cell_kind: str
    region: int
    element: CellGeometryElement
    coordinate_map: tuple[Expression, ...] | None


@dataclass(frozen=True, slots=True)
class _ChildCell:
    row: int
    target_cell_id: int
    parent_cell_id: int
    target_kind: str
    root_kind: str
    region: int
    offset: tuple[Fraction, ...]
    matrix: tuple[tuple[Fraction, ...], ...]
    vertices: tuple[tuple[Fraction, ...], ...]
    measure: Fraction
    chart_id: str


def _coordinate_expression(value: ExpressionInput, /) -> Expression:
    if isinstance(value, RationalPolynomial):
        return value
    return {monomial: coefficient for monomial, coefficient in value.items()}


def _coordinate_expression_row(
    values: tuple[ExpressionInput, ...], /
) -> tuple[Expression, ...]:
    return tuple(_coordinate_expression(value) for value in values)


def _fractional_chart_map(
    value: tuple[tuple[ExactScalar, ...], tuple[tuple[ExactScalar, ...], ...]],
    /,
) -> tuple[tuple[Fraction, ...], tuple[tuple[Fraction, ...], ...]] | None:
    offset, matrix = value
    exact_offset: list[Fraction] = []
    for coefficient in offset:
        if not isinstance(coefficient, Fraction):
            return None
        exact_offset.append(coefficient)
    exact_rows: list[tuple[Fraction, ...]] = []
    for row in matrix:
        exact_row: list[Fraction] = []
        for coefficient in row:
            if not isinstance(coefficient, Fraction):
                return None
            exact_row.append(coefficient)
        exact_rows.append(tuple(exact_row))
    return tuple(exact_offset), tuple(exact_rows)


def _standard_kind(kind: str, dimension: int, /) -> str | None:
    if dimension == 2 and kind in ("triangle", "simplex:2"):
        return "triangle"
    if dimension == 2 and kind == "quadrilateral":
        return "quadrilateral"
    if dimension == 3 and kind in ("tetrahedron", "simplex:3"):
        return "tetrahedron"
    if dimension == 3 and kind in ("hexahedron", "prism", "pyramid"):
        return kind
    return None


def _box_bounds(
    vertices: tuple[tuple[Fraction, ...], ...], dimension: int, /
) -> tuple[tuple[Fraction, Fraction], ...]:
    return tuple(
        (min(point[axis] for point in vertices), max(point[axis] for point in vertices))
        for axis in range(dimension)
    )


def _inside_reference_cell(point: tuple[Fraction, ...], cell_kind: str, /) -> bool:
    if cell_kind == "triangle":
        return all(value >= 0 for value in point) and sum(point, Fraction(0)) <= 1
    if cell_kind == "tetrahedron":
        return all(value >= 0 for value in point) and sum(point, Fraction(0)) <= 1
    if cell_kind in ("quadrilateral", "hexahedron"):
        return all(Fraction(0) <= value <= Fraction(1) for value in point)
    if cell_kind == "prism":
        first, second, station = point
        return first >= 0 and second >= 0 and first + second <= 1 and 0 <= station <= 1
    if cell_kind == "pyramid":
        first, second, height = point
        return (
            0 <= height <= 1
            and height / 2 <= first <= 1 - height / 2
            and height / 2 <= second <= 1 - height / 2
        )
    return False


def _chart_vertices(
    kind: str,
    offset: tuple[Fraction, ...],
    matrix: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    reference = reference_cell_topology(kind)
    reference_points = tuple(
        tuple(Fraction(value) for value in point) for point in reference.vertices
    )
    return tuple(
        tuple(
            offset[row]
            + sum(
                (matrix[row][column] * point[column] for column in range(len(point))),
                Fraction(0),
            )
            for row in range(len(offset))
        )
        for point in reference_points
    )


def _cell_simplices(
    cell_kind: str, vertices: tuple[tuple[Fraction, ...], ...], /
) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
    match cell_kind:
        case "triangle" | "tetrahedron":
            return (vertices,)
        case "quadrilateral":
            routes = ((0, 1, 2), (0, 2, 3))
        case "prism":
            routes = ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5))
        case "pyramid":
            routes = ((0, 1, 2, 4), (0, 2, 3, 4))
        case "hexahedron":
            routes = (
                (0, 1, 2, 6),
                (0, 2, 3, 6),
                (0, 3, 7, 6),
                (0, 7, 4, 6),
                (0, 4, 5, 6),
                (0, 5, 1, 6),
            )
        case _:
            raise ValueError(f"Unsupported exact chart cell {cell_kind!r}.")
    return tuple(tuple(vertices[index] for index in route) for route in routes)


def _positive_simplex_intersection(
    first: tuple[tuple[Fraction, ...], ...],
    second: tuple[tuple[Fraction, ...], ...],
    dimension: int,
    /,
) -> bool:
    if dimension == 2:
        if any(len(point) != 2 for point in (*first, *second)):
            return True
        first_2d = tuple((point[0], point[1]) for point in first)
        second_2d = tuple((point[0], point[1]) for point in second)
        return area2(intersection(first_2d, second_2d)) > 0
    if (
        dimension != 3
        or len(first) != 4
        or len(second) != 4
        or any(len(point) != 3 for point in (*first, *second))
    ):
        return True
    first_3d = tuple((point[0], point[1], point[2]) for point in first)
    second_3d = tuple((point[0], point[1], point[2]) for point in second)
    points = tetrahedron_contact_points(first_3d, second_3d)
    for selected in combinations(points, 4):
        matrix = tuple(
            tuple(selected[row + 1][axis] - selected[0][axis] for axis in range(3))
            for row in range(3)
        )
        if determinant_exact(matrix):
            return True
    return False


def _overlap_interior(first: _ChildCell, second: _ChildCell, dimension: int, /) -> bool:
    first_simplices = _cell_simplices(first.target_kind, first.vertices)
    second_simplices = _cell_simplices(second.target_kind, second.vertices)
    for first_simplex in first_simplices:
        first_bounds = _box_bounds(first_simplex, dimension)
        for second_simplex in second_simplices:
            second_bounds = _box_bounds(second_simplex, dimension)
            if any(
                min(first_bounds[axis][1], second_bounds[axis][1])
                <= max(first_bounds[axis][0], second_bounds[axis][0])
                for axis in range(dimension)
            ):
                continue
            if _positive_simplex_intersection(first_simplex, second_simplex, dimension):
                return True
    return False


def _overlapping_child_pairs(
    children: list[_ChildCell], dimension: int, /
) -> tuple[tuple[int, int], ...]:
    """Find every positive-volume overlap, retaining original pair order."""
    if len(children) < 2:
        return ()
    bounds = tuple(_box_bounds(child.vertices, dimension) for child in children)
    try:
        lower = np.asarray(
            [[_float_down(interval[0]) for interval in box] for box in bounds],
            dtype=np.float64,
        )
        upper = np.asarray(
            [[_float_up(interval[1]) for interval in box] for box in bounds],
            dtype=np.float64,
        )
        finite = bool(
            np.all(np.isfinite(lower))
            and np.all(np.isfinite(upper))
            and np.all(np.abs(lower) <= np.finfo(np.float64).max / 2)
            and np.all(np.abs(upper) <= np.finfo(np.float64).max / 2)
        )
    except OverflowError:
        finite = False
    if finite:
        tree = prepare_bvh(lower, upper, dtype=np.float64)
        candidates: Iterable[tuple[int, int]] = (
            (int(first), int(second))
            for first_ids, second_ids in bvh_overlap_pair_blocks(
                tree, tree, include_touching=True
            )
            for first, second in zip(first_ids, second_ids, strict=True)
            if first < second
        )
    else:
        # Unrepresentable outward boxes cannot exclude any exact pair.
        candidates = combinations(range(len(children)), 2)
    overlaps = []
    for first, second in candidates:
        if any(
            min(bounds[first][axis][1], bounds[second][axis][1])
            <= max(bounds[first][axis][0], bounds[second][axis][0])
            for axis in range(dimension)
        ):
            continue
        if _overlap_interior(children[first], children[second], dimension):
            overlaps.append((first, second))
    return tuple(sorted(overlaps))


def _finding(
    check: str,
    status: MeshFindingStatus,
    kind: MeshCertificateEntityKind,
    ids: tuple[int, ...] = (),
    /,
    *,
    detail: str,
) -> MeshCertificateFinding:
    return MeshCertificateFinding(check, status, kind, ids, detail=detail)


def _root_boundary_proof(
    domain: MappedReferenceDomain,
) -> tuple[tuple[FacetOverlap, ...], int, tuple[MeshCertificateFinding, ...]]:
    reference_mesh = domain.reference_mesh
    affine = CellGeometrySpec.affine(reference_mesh)
    expressions = cell_coordinate_expressions(reference_mesh, affine)
    target_facets, issues = oriented_target_facets(
        reference_mesh,
        np.asarray(domain.cell_regions, dtype=np.int64),
        expressions,
        source_domain=domain.reference_domain,
        geometry=affine,
    )
    findings = [
        _finding(
            issue.check, issue.status, "facet", issue.entity_ids, detail=issue.detail
        )
        for issue in issues
    ]
    overlaps, facet_findings = partition_overlaps(
        domain.reference_domain, target_facets, coordinate_space="reference"
    )
    partition_status: MeshFindingStatus = (
        "unresolved"
        if any(issue.status == "unresolved" for issue in issues)
        else "violated"
    )
    for check, first, second in facet_findings:
        findings.append(
            _finding(
                check,
                partition_status,
                "source_facet"
                if check
                in (
                    "overlapping_source_facets",
                    "uncovered_boundary",
                    "double_or_partial_source_coverage",
                )
                else "facet",
                (first,) if second < 0 else (first, second),
                detail="The reference root mesh boundary does not reproduce the declared source partition.",
            )
        )
    root_measures = domain.reference_domain.exact_region_measures()
    achieved: list[ExactScalar] = [Fraction(0) for _ in domain.region_ids]
    cursor = 0
    for block in reference_mesh.blocks:
        for row in range(block.cell_count):
            try:
                root_expression = expressions[cursor]
                if root_expression is None:
                    raise ValueError("Reference root cell has no exact coordinate map.")
                measure = affine_cell_measure(root_expression, block.cell_kind)
            except (TypeError, ValueError, ZeroDivisionError) as error:
                findings.append(
                    _finding(
                        "reference_root_measure_unresolved",
                        "unresolved",
                        "source_cell",
                        (int(np.asarray(block.global_ids, dtype=np.int64)[row]),),
                        detail=f"Exact root-cell measure is unavailable: {error}.",
                    )
                )
            else:
                if measure <= 0:
                    findings.append(
                        _finding(
                            "reference_root_orientation",
                            "violated",
                            "source_cell",
                            (int(np.asarray(block.global_ids, dtype=np.int64)[row]),),
                            detail="A reference root cell has nonpositive oriented measure.",
                        )
                    )
                else:
                    achieved[int(domain.cell_regions[cursor])] += measure
            cursor += 1
    for region, (expected, actual) in enumerate(
        zip(root_measures, achieved, strict=True)
    ):
        if actual != expected:
            findings.append(
                _finding(
                    "reference_root_region_measure",
                    "violated",
                    "source_region",
                    (region,),
                    detail="The oriented reference-root cell measures differ from the declared source region measure.",
                )
            )
    return overlaps, len(target_facets), tuple(findings)


def _affine_image(
    offset: tuple[ExactScalar, ...],
    matrix: tuple[tuple[ExactScalar, ...], ...],
    point: tuple[Fraction, ...],
    /,
) -> tuple[ExactScalar, ...]:
    return tuple(
        offset[row]
        + sum(
            (matrix[row][column] * point[column] for column in range(len(point))),
            Fraction(0),
        )
        for row in range(len(offset))
    )


def _certify_polyhedral_child_charts(
    domain: MappedReferenceDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source_map: MappedPowerCellGeometrySource,
    cell_regions: np.ndarray,
    root_overlaps: tuple[FacetOverlap, ...],
    root_target_facet_count: int,
    findings: list[MeshCertificateFinding],
    /,
) -> MappedChartProof:
    """Restrict each carrier root chart to the power pieces of one polyhedron.

    Every restricted piece lies in one carrier tetrahedron whose four corners
    have exact root-reference images, so piece coordinates are affine in the
    carrier tetrahedron. A polyhedron whose pieces all restrict one root chart
    has that affine chart; its physical vertices must be the affine root-map
    images of the chart vertices, which makes the published polyhedron the
    exact root image of a reference polyhedron. Containment and exact measure
    completeness are proved here. All children of one root share one affine
    chart, so their pairwise interior disjointness is the chart preimage of the
    certified global embedding that coverage requires before it uses charts.
    """
    connectivity = mesh.connectivity
    reference = source_map.reference_source
    if (
        not isinstance(connectivity, PolyhedralConnectivity)
        or source_map.domain_id != domain.domain_id
        or geometry.source_mesh_id != mesh.mesh_id
        or geometry.source_map_id != source_map.source_map_id
        or any(block.cell_kind != "polyhedron" for block in mesh.blocks)
    ):
        findings.append(
            _finding(
                "mapped_chart_source_identity",
                "violated",
                "mesh",
                detail="Polyhedral chart ancestry names a different mesh, map, or root domain.",
            )
        )
        return MappedChartProof(
            (), (), root_overlaps, tuple(findings), root_target_facet_count
        )
    root_maps = cell_coordinate_expressions(domain.reference_mesh, domain.source_geometry)
    root_kinds = tuple(
        block.cell_kind
        for block in domain.reference_mesh.blocks
        for _ in range(block.cell_count)
    )
    root_rows = {
        int(identifier): row for row, identifier in enumerate(domain.root_cell_ids)
    }
    physical_vertices = geometry.source_coordinates()
    cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    vertex_offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int64)
    vertex_values = np.asarray(connectivity.cell_vertex_values, dtype=np.int64)
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    cell_face_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    face_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int64)
    piece_groups = source_map.cell_piece_tetrahedra()
    if (
        len(reference.source_coordinates()) != len(physical_vertices)
        or cell_regions.shape != (cell_ids.size,)
        or len(piece_groups) != cell_ids.size
    ):
        findings.append(
            _finding(
                "mapped_chart_parent_assignment",
                "unresolved",
                "mesh",
                detail="Reference vertices, physical vertices, and restricted pieces do not align with the target cells.",
            )
        )
        return MappedChartProof(
            (), (), root_overlaps, tuple(findings), root_target_facet_count
        )

    parent_ids: list[int] = []
    chart_ids: list[str] = []
    measures_by_root: dict[int, Fraction] = {}
    for cell_row, target_cell_id in enumerate(cell_ids.tolist()):
        cell_vertices = vertex_values[
            vertex_offsets[cell_row] : vertex_offsets[cell_row + 1]
        ]
        try:
            parent_id, chart_points = source_map.restricted_root_chart(
                piece_groups[cell_row], cell_vertices
            )
        except ValueError as error:
            findings.append(
                _finding(
                    "mapped_chart_parent_assignment",
                    "unresolved",
                    "cell",
                    (target_cell_id,),
                    detail=f"The restricted carrier root chart is unavailable: {error}",
                )
            )
            continue
        root_row = root_rows.get(parent_id)
        if root_row is None:
            findings.append(
                _finding(
                    "mapped_chart_parent_missing",
                    "violated",
                    "source_cell",
                    (parent_id,),
                    detail="A child chart names a root cell absent from the mapped source mesh.",
                )
            )
            continue
        root_kind = _standard_kind(root_kinds[root_row], 3)
        root_map = root_maps[root_row]
        root_affine = (
            None if root_map is None or root_kind is None else affine_map(root_map, 3)
        )
        if root_kind is None or root_affine is None:
            findings.append(
                _finding(
                    "mapped_chart_kind_unsupported",
                    "unresolved",
                    "source_cell",
                    (parent_id,),
                    detail="Planar polyhedral children require an affine map on a supported root cell.",
                )
            )
            continue
        if int(domain.cell_regions[root_row]) != int(cell_regions[cell_row]):
            findings.append(
                _finding(
                    "mapped_chart_region_assignment",
                    "violated",
                    "cell",
                    (target_cell_id, parent_id),
                    detail="Dense target cell region differs from its mapped root cell region.",
                )
            )
        vertex_charts = dict(zip(cell_vertices.tolist(), chart_points, strict=True))
        if not all(
            _inside_reference_cell(point, root_kind) for point in vertex_charts.values()
        ):
            findings.append(
                _finding(
                    "mapped_chart_containment",
                    "violated",
                    "cell",
                    (target_cell_id, parent_id),
                    detail="A polyhedral chart vertex lies outside its convex root reference cell.",
                )
            )
        offset, matrix = root_affine
        if any(
            _affine_image(offset, matrix, point) != tuple(physical_vertices[vertex])
            for vertex, point in vertex_charts.items()
        ):
            findings.append(
                _finding(
                    "mapped_chart_physical_map_mismatch",
                    "violated",
                    "cell",
                    (target_cell_id, parent_id),
                    detail="A physical polyhedral vertex is not the exact root-map image of its chart vertex.",
                )
            )
        measure = Fraction(0)
        start, stop = (
            int(cell_face_offsets[cell_row]),
            int(cell_face_offsets[cell_row + 1]),
        )
        for slot in range(start, stop):
            face = int(cell_faces[slot])
            loop = tuple(
                vertex_charts[int(vertex)]
                for vertex in face_values[face_offsets[face] : face_offsets[face + 1]]
            )
            anchor = loop[0]
            oriented = Fraction(0)
            for index in range(1, len(loop) - 1):
                first, second = loop[index], loop[index + 1]
                oriented += (
                    anchor[0] * (first[1] * second[2] - first[2] * second[1])
                    - anchor[1] * (first[0] * second[2] - first[2] * second[0])
                    + anchor[2] * (first[0] * second[1] - first[1] * second[0])
                ) / 6
            measure += int(face_signs[slot]) * oriented
        if measure <= 0:
            findings.append(
                _finding(
                    "mapped_chart_orientation",
                    "violated",
                    "cell",
                    (target_cell_id,),
                    detail="A polyhedral chart has nonpositive exact root-reference measure.",
                )
            )
            continue
        measures_by_root[parent_id] = (
            measures_by_root.get(parent_id, Fraction(0)) + measure
        )
        parent_ids.append(parent_id)
        chart_ids.append(
            canonical_fingerprint(
                {
                    "parent": parent_id,
                    "target": target_cell_id,
                    "kind": "polyhedron",
                    "vertices": tuple(
                        (vertex, tuple(str(value) for value in point))
                        for vertex, point in sorted(vertex_charts.items())
                    ),
                }
            )
        )
    for parent_id, root_row in root_rows.items():
        achieved = measures_by_root.get(parent_id)
        if achieved is None:
            findings.append(
                _finding(
                    "mapped_chart_root_uncovered",
                    "violated",
                    "source_cell",
                    (parent_id,),
                    detail="No target child chart covers this mapped source root cell.",
                )
            )
            continue
        root_kind = _standard_kind(root_kinds[root_row], 3)
        if root_kind is not None and achieved != reference_cell_measure(root_kind):
            findings.append(
                _finding(
                    "mapped_chart_completeness",
                    "violated",
                    "source_cell",
                    (parent_id,),
                    detail="Contained polyhedral child chart measures do not fill the exact root reference cell.",
                )
            )
    if len(parent_ids) != cell_ids.size:
        findings.append(
            _finding(
                "mapped_chart_boundary_unresolved",
                "unresolved",
                "mesh",
                detail="The complete root-to-target chart partition is unavailable for every target cell.",
            )
        )
    return MappedChartProof(
        tuple(parent_ids),
        tuple(chart_ids),
        root_overlaps,
        tuple(findings),
        root_target_facet_count,
    )


def _polynomial_identity(
    polynomial: Mapping[tuple[int, ...], ExactScalar], /
) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(item)) for key, item in polynomial.items()))


def _expression_identity(value: Expression, /) -> object:
    if isinstance(value, RationalPolynomial):
        return (
            _polynomial_identity(value.numerator),
            _polynomial_identity(value.denominator),
        )
    return _polynomial_identity(value)


def _certify_intermediate_parent_chain(
    domain: MappedReferenceDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_regions: np.ndarray,
    /,
    *,
    chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
    _validated_parent_embedding: ValidatedGlobalEmbedding | None = None,
) -> MappedChartProof | None:
    """Certify a child of a mapped child through its parent's own root proof.

    The restriction names an intermediate parent bound by the geometry's
    retained source map, and that parent's current restriction names the
    mapped root. The parent's chart proof partitions every root; the forward
    reference chain proves the children partition every parent (degree one,
    exact child-in-parent charts); the child-to-root charts are the exact
    compositions of both witness banks, checked against the target physical
    maps. ``None`` means the operands do not form such a two-level chain.
    """
    from ..discretization._cell_geometry_validity import (
        CellValidityPolicy,
        certify_cell_geometry_validity,
    )
    from ._global_embedding import ValidatedGlobalEmbedding
    from ._reference_chain_embedding import certify_reference_chain_embedding

    restriction = geometry.restriction_source
    source_map = geometry.source_map
    if chart_parent is None:
        parent_mesh = getattr(source_map, "source_mesh", None)
        parent_geometry = getattr(source_map, "source_geometry", None)
    else:
        parent_mesh, parent_geometry = chart_parent
    if (
        restriction is None
        or not isinstance(parent_mesh, CellMesh)
        or not isinstance(parent_geometry, CellGeometrySpec)
        or restriction.source_geometry_id != cell_geometry_id(parent_geometry)
        or restriction.source_topology_id != parent_mesh.topology_id
        or getattr(source_map, "target_mesh_id", mesh.mesh_id) != mesh.mesh_id
        or getattr(source_map, "target_topology_id", mesh.topology_id) != mesh.topology_id
    ):
        return None
    if chart_parent is not None:
        from ._mesh_certificates import _same_tree_content
        from ._surface_source_archive import _geometry

        try:
            rebuilt_mesh = CellMesh.from_stored_fields(
                parent_mesh,
                actual_geometry=parent_geometry
                if parent_mesh.periodic_topology is not None
                else None,
            )
            rebuilt_geometry = _geometry(parent_geometry, rebuilt_mesh)
        except (TypeError, ValueError, IndexError):
            return None
        if not _same_tree_content(parent_mesh, rebuilt_mesh) or not _same_tree_content(
            parent_geometry, rebuilt_geometry
        ):
            return None
    parent_restriction = parent_geometry.restriction_source
    if (
        parent_restriction is None
        or parent_restriction.source_geometry_id != domain.reference_geometry_id
        or parent_restriction.source_topology_id != domain.reference_mesh.topology_id
        or set(parent_restriction.block_names)
        != {block.name for block in parent_mesh.blocks}
        or set(restriction.block_names) != {block.name for block in mesh.blocks}
    ):
        return None

    root_regions: dict[int, int] = {}
    root_maps = cell_coordinate_expressions(domain.reference_mesh, domain.source_geometry)
    root_map_by_id: dict[int, tuple[Expression, ...] | None] = {}
    cursor = 0
    for block in domain.reference_mesh.blocks:
        for identifier in np.asarray(block.global_ids, dtype=np.int64).tolist():
            root_regions[int(identifier)] = int(domain.cell_regions[cursor])
            root_map_by_id[int(identifier)] = root_maps[cursor]
            cursor += 1

    parent_index = {name: row for row, name in enumerate(parent_restriction.block_names)}
    parent_rows: dict[int, tuple[int, tuple[Expression, ...] | None]] = {}
    parent_regions: list[int] = []
    for block in parent_mesh.blocks:
        index = parent_index[block.name]
        roots = np.asarray(parent_restriction.parent_cell_ids[index], dtype=np.int64)
        witness = parent_restriction.parent_reference_maps[index]
        charts = None if witness is None else witness.source_reference_chart_arguments
        if roots.shape != (block.cell_count,) or (
            charts is not None and len(charts) != block.cell_count
        ):
            return None
        for row, identifier in enumerate(
            np.asarray(block.global_ids, dtype=np.int64).tolist()
        ):
            root_id = int(roots[row])
            if root_id not in root_regions:
                return None
            parent_rows[int(identifier)] = (
                root_id,
                None if charts is None else _coordinate_expression_row(charts[row]),
            )
            parent_regions.append(root_regions[root_id])

    parent_proof = certify_child_chart_partition(
        domain,
        parent_mesh,
        parent_geometry,
        np.asarray(parent_regions, dtype=np.int64),
    )
    findings = list(parent_proof.findings)
    reference_maps = restriction.block_parent_reference_maps
    chain_id = ""
    try:
        if _validated_parent_embedding is None:
            parent_validity = certify_cell_geometry_validity(
                parent_geometry, mesh=parent_mesh
            )
            parent_embedding = ValidatedGlobalEmbedding(
                parent_mesh, parent_geometry, parent_validity
            )
        else:
            if type(_validated_parent_embedding) is not ValidatedGlobalEmbedding:
                raise TypeError("Mapped parent requires a validated embedding proof.")
            parent_embedding = _validated_parent_embedding
            parent_validity = parent_embedding.validity
            if parent_validity.policy_id != CellValidityPolicy().policy_id:
                raise ValueError("Mapped parent validity uses different proof controls.")
            parent_embedding.require(parent_mesh, parent_geometry, parent_validity)
        chain = certify_reference_chain_embedding(
            parent_mesh,
            parent_geometry,
            mesh,
            geometry,
            reference_maps,
            parent_embedding.certificate,
            _validated_parent_embedding=parent_embedding,
        )
    except (TypeError, ValueError, IndexError, ZeroDivisionError) as error:
        findings.append(
            _finding(
                "mapped_chart_parent_partition",
                "unresolved",
                "mesh",
                detail=f"Children do not have an exact partition proof of their parents: {error}.",
            )
        )
    else:
        chain_id = chain.certificate_id
        if (
            chain.status != "certified"
            or chain.findings
            or not chain.cross_parent_coverage_complete
        ):
            findings.append(
                _finding(
                    "mapped_chart_parent_partition",
                    "violated" if chain.status == "violated" else "unresolved",
                    "mesh",
                    detail="The forward reference chain does not prove the children partition their parents.",
                )
            )

    target_maps = cell_coordinate_expressions(mesh, geometry)
    target_root_ids: list[int] = []
    chart_ids: list[str] = []
    cursor = 0
    for block in mesh.blocks:
        witness = reference_maps.get(block.name)
        charts = None if witness is None else witness.source_reference_chart_arguments
        if witness is None or charts is None:
            findings.append(
                _finding(
                    "mapped_chart_reference_map",
                    "unresolved",
                    "cell",
                    tuple(int(value) for value in np.asarray(block.global_ids).tolist()),
                    detail="A two-level child block lacks exact child-in-parent charts.",
                )
            )
            cursor += block.cell_count
            continue
        parents = np.asarray(witness.source_cell_ids, dtype=np.int64)
        for row, target_id in enumerate(
            np.asarray(block.global_ids, dtype=np.int64).tolist()
        ):
            parent_id = int(parents[row])
            parent = parent_rows.get(parent_id)
            if parent is None:
                findings.append(
                    _finding(
                        "mapped_chart_parent_missing",
                        "violated",
                        "cell",
                        (int(target_id), parent_id),
                        detail="A child chart names a parent cell absent from its parent mesh.",
                    )
                )
                cursor += 1
                continue
            root_id, parent_chart = parent
            if int(cell_regions[cursor]) != root_regions[root_id]:
                findings.append(
                    _finding(
                        "mapped_chart_region_assignment",
                        "violated",
                        "cell",
                        (int(target_id), root_id),
                        detail="Dense target cell region differs from its mapped root cell region.",
                    )
                )
            child_chart = _coordinate_expression_row(charts[row])
            root_map = root_map_by_id[root_id]
            target_map = target_maps[cursor]
            if parent_chart is None or root_map is None or target_map is None:
                findings.append(
                    _finding(
                        "mapped_chart_expression_missing",
                        "unresolved",
                        "cell",
                        (int(target_id),),
                        detail="The exact composed child-to-root chart is unavailable.",
                    )
                )
                cursor += 1
                continue
            composed = tuple(
                compose_expression(expression, child_chart) for expression in parent_chart
            )
            expected = tuple(
                compose_expression(expression, composed) for expression in root_map
            )
            if len(expected) != len(target_map) or any(
                not expression_equal(left, right)
                for left, right in zip(expected, target_map, strict=True)
            ):
                findings.append(
                    _finding(
                        "mapped_chart_physical_map_mismatch",
                        "violated",
                        "cell",
                        (int(target_id), root_id),
                        detail="The target physical coordinate map is not the exact root map composed with its two-level chart.",
                    )
                )
            target_root_ids.append(root_id)
            chart_ids.append(
                canonical_fingerprint(
                    {
                        "root": root_id,
                        "parent": parent_id,
                        "target": int(target_id),
                        "kind": block.cell_kind,
                        "parent_chart_proof": parent_proof.chart_ids,
                        "parent_partition": chain_id,
                        "chart": tuple(_expression_identity(value) for value in composed),
                    }
                )
            )
            cursor += 1
    if len(target_root_ids) != len(cell_regions):
        findings.append(
            _finding(
                "mapped_chart_boundary_unresolved",
                "unresolved",
                "mesh",
                detail="The complete root-to-target chart partition is unavailable for every target cell.",
            )
        )
    return MappedChartProof(
        tuple(target_root_ids),
        tuple(chart_ids),
        parent_proof.facet_overlaps,
        tuple(findings),
        parent_proof.target_facet_count,
    )


def certify_child_chart_partition(
    domain: MappedReferenceDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_regions: np.ndarray,
    /,
    *,
    chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
    _validated_parent_embedding: ValidatedGlobalEmbedding | None = None,
) -> MappedChartProof:
    """Prove every target chart lies in, partitions, and covers its root cell."""
    dimension = domain.ambient_dimension
    findings: list[MeshCertificateFinding] = []
    if _validated_parent_embedding is not None:
        from ._global_embedding import ValidatedGlobalEmbedding

        restriction = geometry.restriction_source
        if (
            type(_validated_parent_embedding) is not ValidatedGlobalEmbedding
            or chart_parent is None
            or restriction is None
            or (
                restriction.source_geometry_id == domain.reference_geometry_id
                and restriction.source_topology_id == domain.reference_mesh.topology_id
            )
        ):
            raise ValueError("Validated parent embedding requires an intermediate chart parent.")
        _validated_parent_embedding.require(
            chart_parent[0], chart_parent[1], _validated_parent_embedding.validity
        )
    root_overlaps, root_target_facet_count, root_findings = _root_boundary_proof(domain)
    findings.extend(root_findings)
    if isinstance(geometry.source_map, MappedPowerCellGeometrySource):
        return _certify_polyhedral_child_charts(
            domain,
            mesh,
            geometry,
            geometry.source_map,
            np.asarray(cell_regions, dtype=np.int64),
            root_overlaps,
            root_target_facet_count,
            findings,
        )
    identity = (
        mesh.mesh_id == domain.reference_mesh.mesh_id
        and geometry.geometry_layout_id == domain.source_geometry.geometry_layout_id
        and cell_geometry_id(geometry) == domain.reference_geometry_id
    )
    restriction = geometry.restriction_source
    if not identity:
        if restriction is None:
            findings.append(
                _finding(
                    "mapped_chart_ancestry_missing",
                    "unresolved",
                    "mesh",
                    detail="A child mesh needs retained exact reference-map ancestry to its mapped source roots.",
                )
            )
            return MappedChartProof(
                (), (), root_overlaps, tuple(findings), root_target_facet_count
            )
        if (
            restriction.source_geometry_id != domain.reference_geometry_id
            or restriction.source_topology_id != domain.reference_mesh.topology_id
        ):
            two_level = _certify_intermediate_parent_chain(
                domain,
                mesh,
                geometry,
                np.asarray(cell_regions, dtype=np.int64),
                chart_parent=chart_parent,
                _validated_parent_embedding=_validated_parent_embedding,
            )
            if two_level is not None:
                return two_level
            if chart_parent is None and not (
                isinstance(getattr(geometry.source_map, "source_mesh", None), CellMesh)
                and isinstance(
                    getattr(geometry.source_map, "source_geometry", None),
                    CellGeometrySpec,
                )
            ):
                findings.append(
                    _finding(
                        "mapped_chart_parent_premise",
                        "unresolved",
                        "mesh",
                        detail="Child-chart ancestry requires its actual parent mesh and geometry.",
                    )
                )
                return MappedChartProof(
                    (), (), root_overlaps, tuple(findings), root_target_facet_count
                )
            findings.append(
                _finding(
                    "mapped_chart_source_identity",
                    "violated",
                    "mesh",
                    detail="Child-chart ancestry names a different root geometry or topology.",
                )
            )
            return MappedChartProof(
                (), (), root_overlaps, tuple(findings), root_target_facet_count
            )

    root_elements, _, _ = domain.source_geometry.resolve(domain.reference_mesh)
    root_cell_rows: dict[int, _RootCell] = {}
    root_cursor = 0
    root_maps = cell_coordinate_expressions(domain.reference_mesh, domain.source_geometry)
    for block_index, (block, element) in enumerate(
        zip(domain.reference_mesh.blocks, root_elements, strict=True)
    ):
        for block_row, identifier in enumerate(
            np.asarray(block.global_ids, dtype=np.int64)
        ):
            root_map = root_maps[root_cursor]
            root_cell_rows[int(identifier)] = _RootCell(
                block_index,
                block.cell_kind,
                int(domain.cell_regions[root_cursor]),
                element,
                root_map,
            )
            root_cursor += 1

    restriction_parents = {} if restriction is None else restriction.block_parent_cell_ids
    restriction_source_blocks = (
        {} if restriction is None else (restriction.block_source_blocks or {})
    )
    restriction_reference_maps = (
        {} if restriction is None else restriction.block_parent_reference_maps
    )
    target_elements, _, _ = geometry.resolve(mesh)
    target_map_rows = cell_coordinate_expressions(mesh, geometry)
    target_cursor = 0
    child_cells: list[_ChildCell] = []
    target_parent_ids: list[int] = []
    chart_ids: list[str] = []
    for block_index, (block, element) in enumerate(
        zip(mesh.blocks, target_elements, strict=True)
    ):
        parent_bank = None if identity else restriction_parents.get(block.name)
        source_block_name = restriction_source_blocks.get(block.name)
        reference_map = None if identity else restriction_reference_maps.get(block.name)
        reference_chart_bank: tuple[tuple[Expression, ...], ...] | None = None
        reference_map_error = None
        if reference_map is not None:
            try:
                validate_reference_cell_map_witnesses(reference_map)
            except (TypeError, ValueError, IndexError, ArithmeticError) as error:
                reference_map_error = str(error)
            else:
                stored_bank = reference_map.source_reference_chart_arguments
                if stored_bank is not None:
                    reference_chart_bank = tuple(
                        _coordinate_expression_row(chart_row) for chart_row in stored_bank
                    )
        for block_row, target_cell_id in enumerate(
            np.asarray(block.global_ids, dtype=np.int64)
        ):
            row = target_cursor
            region = int(cell_regions[row])
            if not isinstance(element, ScalarCoordinateElement):
                findings.append(
                    _finding(
                        "mapped_chart_coordinate_element",
                        "unresolved",
                        "cell",
                        (int(target_cell_id),),
                        detail="Exact reference charts require a scalar coordinate element.",
                    )
                )
                target_cursor += 1
                continue
            chart: tuple[Expression, ...] | None
            if identity:
                parent_id = int(target_cell_id)
                chart = tuple(
                    {tuple(int(axis == column) for axis in range(dimension)): Fraction(1)}
                    for column in range(dimension)
                )
            else:
                if parent_bank is None or np.asarray(parent_bank).shape != (
                    block.cell_count,
                ):
                    findings.append(
                        _finding(
                            "mapped_chart_parent_assignment",
                            "unresolved",
                            "cell",
                            (int(target_cell_id),),
                            detail="Restriction ancestry does not assign this target cell to one root global cell ID.",
                        )
                    )
                    target_cursor += 1
                    continue
                parent_id = int(np.asarray(parent_bank, dtype=np.int64)[block_row])
                root_row = root_cell_rows.get(parent_id)
                if root_row is None:
                    findings.append(
                        _finding(
                            "mapped_chart_parent_missing",
                            "violated",
                            "source_cell",
                            (parent_id,),
                            detail="A child chart names a root cell absent from the mapped source mesh.",
                        )
                    )
                    target_cursor += 1
                    continue
                if (
                    source_block_name
                    and domain.reference_mesh.blocks[root_row.block_index].name
                    != source_block_name
                ):
                    findings.append(
                        _finding(
                            "mapped_chart_source_block",
                            "violated",
                            "source_cell",
                            (parent_id,),
                            detail="Restriction ancestry names a different root source block.",
                        )
                    )
                    target_cursor += 1
                    continue
                if reference_map_error is not None:
                    findings.append(
                        _finding(
                            "mapped_chart_reference_map",
                            "violated",
                            "cell",
                            (int(target_cell_id),),
                            detail=f"Exact reference-map witness does not revalidate: {reference_map_error}.",
                        )
                    )
                    target_cursor += 1
                    continue
                if reference_map is not None:
                    if (
                        reference_map.target_cell_kind != block.cell_kind
                        or reference_map.source_cell_kind != root_row.cell_kind
                        or not np.array_equal(
                            np.asarray(reference_map.target_cell_ids, dtype=np.int64),
                            np.arange(block.cell_count, dtype=np.int64),
                        )
                        or int(
                            np.asarray(reference_map.source_cell_ids, dtype=np.int64)[
                                block_row
                            ]
                        )
                        != parent_id
                    ):
                        findings.append(
                            _finding(
                                "mapped_chart_reference_binding",
                                "violated",
                                "cell",
                                (int(target_cell_id), parent_id),
                                detail="Exact reference-map rows do not bind this target cell and root parent.",
                            )
                        )
                        target_cursor += 1
                        continue
                chart = (
                    None
                    if reference_chart_bank is None
                    else reference_chart_bank[block_row]
                )
                if chart is None:
                    try:
                        chain = coordinate_reference_chain(element)
                    except (TypeError, ValueError) as error:
                        findings.append(
                            _finding(
                                "mapped_chart_ancestry_unresolved",
                                "unresolved",
                                "cell",
                                (int(target_cell_id),),
                                detail=f"Exact coordinate ancestry cannot be traversed: {error}.",
                            )
                        )
                        target_cursor += 1
                        continue
                    matching_roots = [
                        item
                        for item in chain
                        if item.element_id == root_row.element.element_id
                        and _same_tree_content(item, root_row.element)
                    ]
                    if not matching_roots:
                        findings.append(
                            _finding(
                                "mapped_chart_source_element",
                                "violated",
                                "cell",
                                (int(target_cell_id),),
                                detail="The target coordinate element does not descend from the declared mapped root element.",
                            )
                        )
                        target_cursor += 1
                        continue
                    try:
                        raw_chart = restriction_chart_arguments(
                            element,
                            ancestor=matching_roots[-1],
                            cell_row=block_row,
                        )
                        if raw_chart is None and len(chain) == 1:
                            # Same-element ancestry supplies an identity candidate;
                            # the physical composition is still checked below.
                            chart = tuple(
                                {
                                    tuple(
                                        int(axis == column) for axis in range(dimension)
                                    ): Fraction(1)
                                }
                                for column in range(dimension)
                            )
                        else:
                            chart = (
                                None
                                if raw_chart is None
                                else _coordinate_expression_row(raw_chart)
                            )
                    except (TypeError, ValueError, IndexError) as error:
                        findings.append(
                            _finding(
                                "mapped_chart_expression_unresolved",
                                "unresolved",
                                "cell",
                                (int(target_cell_id),),
                                detail=f"Exact target-to-root chart expressions are unavailable: {error}.",
                            )
                        )
                        target_cursor += 1
                        continue
                if chart is None:
                    findings.append(
                        _finding(
                            "mapped_chart_expression_missing",
                            "unresolved",
                            "cell",
                            (int(target_cell_id),),
                            detail="Target-to-root chart ancestry has no exact coordinate map.",
                        )
                    )
                    target_cursor += 1
                    continue
            target_kind = _standard_kind(block.cell_kind, dimension)
            root_row = root_cell_rows.get(parent_id)
            if root_row is None:
                findings.append(
                    _finding(
                        "mapped_chart_parent_missing",
                        "violated",
                        "source_cell",
                        (parent_id,),
                        detail="A child chart names a root cell absent from the mapped source mesh.",
                    )
                )
                target_cursor += 1
                continue
            root_map = root_row.coordinate_map
            if root_map is None:
                findings.append(
                    _finding(
                        "mapped_chart_root_coordinate_map",
                        "unresolved",
                        "source_cell",
                        (parent_id,),
                        detail="The mapped root cell has no exact physical coordinate map.",
                    )
                )
                target_cursor += 1
                continue
            root_kind = _standard_kind(root_row.cell_kind, dimension)
            chart_affine = (
                None
                if target_kind is None or root_kind is None
                else affine_map(tuple(chart), dimension)
            )
            if target_kind is None or root_kind is None or chart_affine is None:
                findings.append(
                    _finding(
                        "mapped_chart_kind_unsupported",
                        "unresolved",
                        "cell",
                        (int(target_cell_id),),
                        detail="Exact chart coverage requires affine maps over supported simplex, tensor, prism, or pyramid reference cells.",
                    )
                )
                target_cursor += 1
                continue
            rational_chart = _fractional_chart_map(chart_affine)
            if rational_chart is None:
                findings.append(
                    _finding(
                        "mapped_chart_coefficient_field",
                        "unresolved",
                        "cell",
                        (int(target_cell_id),),
                        detail="Child chart coefficients are not rational exact values.",
                    )
                )
                target_cursor += 1
                continue
            offset, matrix = rational_chart
            det = determinant_exact(matrix)
            if det <= 0:
                findings.append(
                    _finding(
                        "mapped_chart_orientation",
                        "violated",
                        "cell",
                        (int(target_cell_id),),
                        detail="Target-to-root chart has nonpositive exact orientation.",
                    )
                )
                target_cursor += 1
                continue
            if root_row.region != region:
                findings.append(
                    _finding(
                        "mapped_chart_region_assignment",
                        "violated",
                        "cell",
                        (int(target_cell_id), parent_id),
                        detail="Dense target cell region differs from its mapped root cell region.",
                    )
                )
            chart_vertices = _chart_vertices(target_kind, offset, matrix)
            contained = all(
                _inside_reference_cell(point, root_kind) for point in chart_vertices
            )
            if not contained:
                findings.append(
                    _finding(
                        "mapped_chart_containment",
                        "violated",
                        "cell",
                        (int(target_cell_id), parent_id),
                        detail="A target chart vertex lies outside its convex root reference cell.",
                    )
                )
            try:
                reference_measure = reference_cell_measure(target_kind)
            except (TypeError, ValueError) as error:
                findings.append(
                    _finding(
                        "mapped_chart_measure_unresolved",
                        "unresolved",
                        "cell",
                        (int(target_cell_id),),
                        detail=f"Exact target chart measure is unavailable: {error}.",
                    )
                )
                target_cursor += 1
                continue
            if not isinstance(reference_measure, Fraction):
                findings.append(
                    _finding(
                        "mapped_chart_measure_unresolved",
                        "unresolved",
                        "cell",
                        (int(target_cell_id),),
                        detail="Exact reference-cell measure is not rational.",
                    )
                )
                target_cursor += 1
                continue
            chart_measure = det * reference_measure
            chart_id = canonical_fingerprint(
                {
                    "parent": parent_id,
                    "target": int(target_cell_id),
                    "kind": target_kind,
                    "offset": tuple(str(value) for value in offset),
                    "matrix": tuple(
                        tuple(str(value) for value in line) for line in matrix
                    ),
                }
            )
            child = _ChildCell(
                row,
                int(target_cell_id),
                parent_id,
                target_kind,
                root_kind,
                region,
                offset,
                matrix,
                chart_vertices,
                chart_measure,
                chart_id,
            )
            target_map = target_map_rows[row]
            if target_map is None:
                findings.append(
                    _finding(
                        "mapped_chart_target_coordinate_map",
                        "unresolved",
                        "cell",
                        (int(target_cell_id),),
                        detail="The target cell has no exact physical coordinate map.",
                    )
                )
                target_cursor += 1
                continue
            expected_map = tuple(
                compose_expression(expression, chart) for expression in root_map
            )
            if len(expected_map) != len(target_map) or any(
                not expression_equal(expected, actual)
                for expected, actual in zip(expected_map, target_map, strict=True)
            ):
                findings.append(
                    _finding(
                        "mapped_chart_physical_map_mismatch",
                        "violated",
                        "cell",
                        (int(target_cell_id), parent_id),
                        detail="The target physical coordinate map is not the exact root map composed with its retained chart.",
                    )
                )
            child_cells.append(child)
            target_parent_ids.append(parent_id)
            chart_ids.append(chart_id)
            target_cursor += 1

    by_parent: dict[int, list[_ChildCell]] = {
        identifier: [] for identifier in root_cell_rows
    }
    for child in child_cells:
        by_parent[child.parent_cell_id].append(child)
    for parent_id, root in root_cell_rows.items():
        children = by_parent[parent_id]
        if not children:
            findings.append(
                _finding(
                    "mapped_chart_root_uncovered",
                    "violated",
                    "source_cell",
                    (parent_id,),
                    detail="No target child chart covers this mapped source root cell.",
                )
            )
            continue
        for first_index, second_index in _overlapping_child_pairs(children, dimension):
            first, second = children[first_index], children[second_index]
            findings.append(
                _finding(
                    "mapped_chart_overlap",
                    "violated",
                    "cell",
                    (first.target_cell_id, second.target_cell_id),
                    detail="Two child charts have a positive-volume intersection inside the same source root cell.",
                )
            )
        try:
            expected = reference_cell_measure(root.cell_kind)
        except (TypeError, ValueError) as error:
            findings.append(
                _finding(
                    "mapped_chart_root_measure_unresolved",
                    "unresolved",
                    "source_cell",
                    (parent_id,),
                    detail=f"Exact root reference measure is unavailable: {error}.",
                )
            )
        else:
            achieved = sum((child.measure for child in children), Fraction(0))
            if achieved != expected:
                findings.append(
                    _finding(
                        "mapped_chart_completeness",
                        "violated",
                        "source_cell",
                        (parent_id,),
                        detail="Contained disjoint child chart measures do not fill the exact root reference cell.",
                    )
                )

    # Every target chart is contained in and disjointly fills its root cell;
    # the root boundary proof therefore transports the complete oriented
    # exterior/interface chain without comparing unrelated local chart frames.
    if len({child.row for child in child_cells}) != len(cell_regions):
        findings.append(
            _finding(
                "mapped_chart_boundary_unresolved",
                "unresolved",
                "mesh",
                detail="The complete root-to-target chart partition is unavailable for every target cell.",
            )
        )
    return MappedChartProof(
        tuple(target_parent_ids),
        tuple(chart_ids),
        root_overlaps,
        tuple(findings),
        root_target_facet_count,
    )


__all__ = ["MappedChartProof", "certify_child_chart_partition"]
