#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-authoritative exact CAD chains and bounded source homotopies.

A carrier complex is never promoted to source authority. Exact planar CAD
chains are expanded from authored surface charts; bounded routes require a
revalidated source-chart correspondence or an interval-regular implicit
stratification tying every output boundary facet to original source roots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Literal, TYPE_CHECKING

import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ._domain_coverage import PiecewiseLinearDomain
from ._global_embedding import GlobalEmbeddingCertificate
from ._meshing_domain import MeshingDomain
from ._planar_primitives import (
    box_overlap_pairs,
    polygon_boxes,
    triangle_pair_separation,
)
from ._source_boundary_records import SourceBoundaryChartWitness, SourceBoundaryQuery
from ._source_fidelity_certificate import SourceFidelityCertificate
from .implicit._adaptive_discovery import (
    AdaptiveImplicitBoundarySource,
    AdaptiveImplicitScaffold,
    AdaptiveImplicitSurface,
)


if TYPE_CHECKING:
    from ._domain_boundary import _RootedJunction
    from ._meshing_domain import PatchCurveUse
    from ._planar_coverage import OrientedFacet
    from ._surface_source_support import _CadCurveUseBinding, SurfaceSourceCharts
    from .brep._curve_approximation_topology import TrimReferenceSegment
    from .implicit._source_charts import (
        AdaptiveImplicitRootChartMap,
        AdaptiveImplicitSourceQuery,
    )

_Point2 = tuple[Fraction, Fraction]
_Triangle2 = tuple[_Point2, _Point2, _Point2]
_Point3 = tuple[Fraction, Fraction, Fraction]
_Triangle3 = tuple[_Point3, _Point3, _Point3]
_GridPoint = tuple[int, int, int]
_VertexKey = tuple[object, ...]


class SourceCoverageUnavailable(ValueError):
    """A source boundary route lacks premises for its coverage theorem."""

    def __init__(
        self, message: str, /, *, status: Literal["violated", "unresolved"]
    ) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True)
class ExactCadSourceBoundary:
    """Exact affine chain realized from a CAD domain's authored planar patches."""

    chain: PiecewiseLinearDomain
    domain_id: str
    source_facet_ancestry: tuple[tuple[int, int], ...]
    boundary_id: str


@dataclass(frozen=True, slots=True)
class BoundedSourceCoverage:
    """Recomputed, oriented source-to-target boundary homotopy evidence."""

    domain_id: str
    source_id: str
    source_revision: str
    region_ids: tuple[str, ...]
    region_occurrence_paths: tuple[tuple[str, ...], ...]
    source_facet_count: int
    target_facet_count: int
    covered_source_facet_count: int
    boundary_degree_ids: tuple[str, ...]
    distance_upper: float
    witness_id: str


@dataclass(frozen=True, slots=True)
class SourceCoverageFailure:
    check: str
    status: Literal["violated", "unresolved"]
    detail: str
    entity_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceCoverageAttempt:
    bounded: BoundedSourceCoverage | None
    failures: tuple[SourceCoverageFailure, ...]


def _point2(values: tuple[object, ...], /) -> _Point2:
    if len(values) != 2:
        raise SourceCoverageUnavailable(
            "A source chart point does not have two exact coordinates.",
            status="unresolved",
        )
    first, second = values
    if not isinstance(first, Fraction) or not isinstance(second, Fraction):
        raise SourceCoverageUnavailable(
            "A source chart point is not exactly rational.", status="unresolved"
        )
    return first, second


def _point3(values: tuple[object, ...], /) -> _Point3:
    if len(values) != 3:
        raise SourceCoverageUnavailable(
            "A source boundary point does not have three exact coordinates.",
            status="unresolved",
        )
    first, second, third = values
    if (
        not isinstance(first, Fraction)
        or not isinstance(second, Fraction)
        or not isinstance(third, Fraction)
    ):
        raise SourceCoverageUnavailable(
            "A source boundary point is not exactly rational.", status="unresolved"
        )
    return first, second, third


def _triangle2(values: tuple[tuple[Fraction, ...], ...], /) -> _Triangle2:
    if len(values) != 3:
        raise SourceCoverageUnavailable(
            "A source chart piece is not triangular.", status="unresolved"
        )
    return (_point2(values[0]), _point2(values[1]), _point2(values[2]))


def _triangle3(values: tuple[tuple[Fraction, ...], ...], /) -> _Triangle3:
    if len(values) != 3:
        raise SourceCoverageUnavailable(
            "A source boundary piece is not triangular.", status="unresolved"
        )
    return (_point3(values[0]), _point3(values[1]), _point3(values[2]))


def _surface_point(
    surface: object, uv: tuple[Fraction, Fraction], /
) -> tuple[Fraction, Fraction, Fraction]:
    from ._meshing_domain import _PlacedSurfacePatch
    from ._rational import rational_vector
    from .brep._patches import ExactLineCurve, ExtrusionSurface, LineCurve, PlanePatch

    if isinstance(surface, _PlacedSurfacePatch):
        placed = surface.placement.apply(_surface_point(surface.source, uv))
        return _point3(tuple(placed))
    if isinstance(surface, PlanePatch):
        origin = rational_vector(np.asarray(surface.origin))
        first = rational_vector(np.asarray(surface.first_axis))
        second = rational_vector(np.asarray(surface.second_axis))
        point = tuple(
            origin[axis] + uv[0] * first[axis] + uv[1] * second[axis] for axis in range(3)
        )
    elif isinstance(surface, ExtrusionSurface) and isinstance(
        surface.curve, ExactLineCurve
    ):
        origin = surface.curve.first_exact
        profile = tuple(
            last - first
            for first, last in zip(
                surface.curve.first_exact, surface.curve.last_exact, strict=True
            )
        )
        direction = rational_vector(np.asarray(surface.direction))
        point = tuple(
            origin[axis] + uv[0] * profile[axis] + uv[1] * direction[axis]
            for axis in range(3)
        )
    elif isinstance(surface, ExtrusionSurface) and isinstance(surface.curve, LineCurve):
        origin = rational_vector(np.asarray(surface.curve.origin))
        profile = rational_vector(np.asarray(surface.curve.direction))
        direction = rational_vector(np.asarray(surface.direction))
        point = tuple(
            origin[axis] + uv[0] * profile[axis] + uv[1] * direction[axis]
            for axis in range(3)
        )
    else:
        raise SourceCoverageUnavailable(
            "Exact CAD boundary coverage requires an affine PlanePatch or extrusion of an authored line; curved patches need a verified chart homotopy.",
            status="unresolved",
        )
    return _point3(point)


def _validate_cad_authority(domain: MeshingDomain, /) -> None:
    from .brep._model import BRepGeometry, BRepModel

    authority = domain.brep_authority
    if authority is None:
        return
    if isinstance(authority, BRepModel):
        rebuilt = MeshingDomain.from_brep(authority)
        if not _same_domain(rebuilt, domain):
            raise SourceCoverageUnavailable(
                "MeshingDomain region boundaries differ from their retained BRepModel authority.",
                status="violated",
            )
        return
    if isinstance(authority, BRepGeometry):
        expected_regions = tuple(
            (
                f"solid_{solid}",
                tuple(
                    (int(face), int(sign))
                    for face, sign in zip(faces, signs, strict=True)
                ),
            )
            for solid, (faces, signs) in enumerate(
                zip(
                    authority.topology.solid_faces,
                    authority.topology.solid_face_orientations,
                    strict=True,
                )
            )
        )
        actual_regions = tuple(
            (region.name, region.boundary) for region in domain.regions
        )
        if actual_regions != expected_regions:
            raise SourceCoverageUnavailable(
                "MeshingDomain region boundaries omit or reorient retained BRepGeometry solids.",
                status="violated",
            )
        return
    raise TypeError("MeshingDomain.brep_authority has an unsupported source owner type.")


def cad_material_sides(
    domain: MeshingDomain, /
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[tuple[int, int], ...], ...]]:
    """Return oriented region pairs and each patch's region-normal signs."""
    _validate_cad_authority(domain)
    rows: list[dict[int, int]] = [dict() for _ in domain.patches]
    for region_row, region in enumerate(domain.regions):
        for patch, orientation in region.boundary:
            if region_row in rows[patch]:
                raise SourceCoverageUnavailable(
                    "A CAD region repeats an authored source patch.", status="violated"
                )
            rows[patch][region_row] = int(orientation)
    from ._meshing_domain import PatchCurveUse

    curve_incidence: dict[tuple[int, int], list[int]] = {}
    for region_row, region in enumerate(domain.regions):
        for patch, orientation in region.boundary:
            source_patch = domain.patches[patch]
            for loop in source_patch.loops:
                for use in loop:
                    if isinstance(use, PatchCurveUse):
                        key = (region_row, use.curve)
                        curve_incidence.setdefault(key, []).append(
                            source_patch.orientation * int(orientation) * use.sense
                        )
    for (region_row, curve), signs in curve_incidence.items():
        if len(signs) != 2 or sum(signs) != 0:
            raise SourceCoverageUnavailable(
                f"CAD region {region_row} has an open or wrongly oriented source curve {curve}.",
                status="violated",
            )
    pairs: list[tuple[int, int]] = []
    signs_by_patch: list[tuple[tuple[int, int], ...]] = []
    for patch, incident in enumerate(domain.patch_regions):
        if set(rows[patch]) != set(incident) or not incident or len(incident) > 2:
            raise SourceCoverageUnavailable(
                f"CAD patch {patch} has missing, unassigned, or nonmanifold material incidence.",
                status="violated",
            )
        signs = tuple(
            (region, domain.patches[patch].orientation * rows[patch][region])
            for region in sorted(incident)
        )
        if len(signs) == 1:
            region, sign = signs[0]
            pair = (region, -1) if sign > 0 else (-1, region)
        else:
            positive = tuple(region for region, sign in signs if sign > 0)
            negative = tuple(region for region, sign in signs if sign < 0)
            if len(positive) != 1 or len(negative) != 1:
                raise SourceCoverageUnavailable(
                    f"CAD interface patch {patch} does not have opposite oriented material sides.",
                    status="violated",
                )
            pair = (positive[0], negative[0])
        pairs.append(pair)
        signs_by_patch.append(signs)
    return tuple(pairs), tuple(signs_by_patch)


def exact_planar_cad_boundary(domain: MeshingDomain, /) -> ExactCadSourceBoundary:
    """Triangulate exact authored planar trims, preserving source patch ancestry."""
    if not isinstance(domain, MeshingDomain):
        raise TypeError("domain must be MeshingDomain.")
    domain.require_current(domain.source_id, domain.source_revision)
    if not domain.regions:
        raise SourceCoverageUnavailable(
            "CAD source has no material regions.", status="unresolved"
        )
    if domain.accuracy != "exact_represented":
        raise SourceCoverageUnavailable(
            "Exact CAD chain extraction requires exact-represented source accuracy.",
            status="unresolved",
        )
    from ._domain_boundary import MeshingDomainBoundarySource

    source = MeshingDomainBoundarySource(domain, tuple(range(len(domain.patches))))
    try:
        patch_triangles = source.boundary_parameter_triangles()
        pairs, _ = cad_material_sides(domain)
    except SourceCoverageUnavailable:
        raise
    except (TypeError, ValueError, IndexError, ArithmeticError) as error:
        raise SourceCoverageUnavailable(
            f"Authored CAD trims do not admit an exact oriented triangulation: {error}",
            status="unresolved",
        ) from error
    if tuple(patch for patch, _ in patch_triangles) != tuple(range(len(domain.patches))):
        raise SourceCoverageUnavailable(
            "The exact CAD trim triangulation omitted a source patch.", status="violated"
        )
    exact_vertices: list[tuple[Fraction, Fraction, Fraction]] = []
    vertex_rows: dict[tuple[Fraction, Fraction, Fraction], int] = {}
    facets: list[tuple[int, int, int]] = []
    facet_regions: list[tuple[int, int]] = []
    ancestry: list[tuple[int, int]] = []
    for patch, triangles in patch_triangles:
        pair = pairs[patch]
        for triangle_row, triangle in enumerate(triangles):
            physical = tuple(
                _surface_point(domain.patches[patch].surface, uv) for uv in triangle
            )
            rows: list[int] = []
            for point in physical:
                row = vertex_rows.get(point)
                if row is None:
                    represented = tuple(float(value) for value in point)
                    if any(
                        Fraction(value) != exact
                        for value, exact in zip(represented, point, strict=True)
                    ):
                        raise SourceCoverageUnavailable(
                            "A CAD trim vertex is not exactly representable for affine target overlap.",
                            status="unresolved",
                        )
                    row = len(exact_vertices)
                    vertex_rows[point] = row
                    exact_vertices.append(point)
                rows.append(row)
            # cad_material_sides already incorporates patch.orientation in its region-side ordering.
            facets.append((rows[0], rows[1], rows[2]))
            facet_regions.append(pair)
            ancestry.append((patch, triangle_row))
    try:
        chain = PiecewiseLinearDomain(
            np.asarray(
                [[float(value) for value in point] for point in exact_vertices],
                dtype=np.float64,
            ),
            np.asarray(facets, dtype=np.int64),
            np.asarray(facet_regions, dtype=np.int64),
            tuple(region.name for region in domain.regions),
            source_id=domain.source_id,
            region_occurrence_paths=domain.region_source_occurrences,
        )
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise SourceCoverageUnavailable(
            f"Authored CAD material boundary is not a closed positively oriented source chain: {error}",
            status="violated",
        ) from error
    boundary_id = canonical_fingerprint(
        {
            "kind": "exact-planar-cad-source-boundary",
            "domain": domain.domain_id,
            "source": (domain.source_id, domain.source_revision, domain.authority_id),
            "regions": tuple(region.name for region in domain.regions),
            "region_occurrence_paths": domain.region_source_occurrences,
            "ancestry": tuple(ancestry),
            "source_chain": chain.domain_id,
        }
    )
    return ExactCadSourceBoundary(chain, domain.domain_id, tuple(ancestry), boundary_id)


def _fraction(value: float, /) -> Fraction:
    return Fraction(float(value))


def _area2(points: tuple[tuple[Fraction, Fraction], ...], /) -> Fraction:
    if len(points) < 3:
        return Fraction(0)
    return (
        sum(
            (
                first[0] * second[1] - first[1] * second[0]
                for first, second in zip(points, (*points[1:], points[0]), strict=True)
            ),
            Fraction(0),
        )
        / 2
    )


def _cross2(
    first: tuple[Fraction, Fraction],
    second: tuple[Fraction, Fraction],
    point: tuple[Fraction, Fraction],
    /,
) -> Fraction:
    return (second[0] - first[0]) * (point[1] - first[1]) - (second[1] - first[1]) * (
        point[0] - first[0]
    )


def _line_intersection(
    first: tuple[Fraction, Fraction],
    second: tuple[Fraction, Fraction],
    line_first: tuple[Fraction, Fraction],
    line_second: tuple[Fraction, Fraction],
    /,
) -> tuple[Fraction, Fraction]:
    direction = (second[0] - first[0], second[1] - first[1])
    line = (line_second[0] - line_first[0], line_second[1] - line_first[1])
    denominator = line[0] * direction[1] - line[1] * direction[0]
    if denominator == 0:
        raise SourceCoverageUnavailable(
            "Exact source-chart clipping is singular.", status="unresolved"
        )
    offset = (line_first[0] - first[0], line_first[1] - first[1])
    parameter = (line[0] * offset[1] - line[1] * offset[0]) / denominator
    return (first[0] + parameter * direction[0], first[1] + parameter * direction[1])


def _intersection_area(
    subject: tuple[tuple[Fraction, Fraction], ...],
    clip: tuple[tuple[Fraction, Fraction], ...],
    /,
) -> Fraction:
    clip_area = _area2(clip)
    if clip_area == 0:
        return Fraction(0)
    if clip_area < 0:
        clip = tuple(reversed(clip))
    polygon = subject
    for line_first, line_second in zip(clip, (*clip[1:], clip[0]), strict=True):
        if not polygon:
            return Fraction(0)
        output: list[tuple[Fraction, Fraction]] = []
        previous = polygon[-1]
        previous_inside = _cross2(line_first, line_second, previous) >= 0
        for current in polygon:
            current_inside = _cross2(line_first, line_second, current) >= 0
            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(previous, current, line_first, line_second)
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    _line_intersection(previous, current, line_first, line_second)
                )
            previous, previous_inside = current, current_inside
        compact: list[tuple[Fraction, Fraction]] = []
        for point in output:
            if not compact or point != compact[-1]:
                compact.append(point)
        if len(compact) > 1 and compact[0] == compact[-1]:
            compact.pop()
        polygon = tuple(compact)
    return abs(_area2(polygon))


_Box2 = tuple[Fraction, Fraction, Fraction, Fraction]


def _triangle_box(triangle: _Triangle2, /) -> _Box2:
    """Exact ``(x_min, x_max, y_min, y_max)`` of a chart triangle."""
    xs = tuple(point[0] for point in triangle)
    ys = tuple(point[1] for point in triangle)
    return min(xs), max(xs), min(ys), max(ys)


def _boxes_overlap(first: _Box2, second: _Box2, /) -> bool:
    """Whether two boxes share positive area; otherwise their triangles meet in area 0."""
    return (
        first[0] < second[1]
        and second[0] < first[1]
        and first[2] < second[3]
        and second[2] < first[3]
    )


def _possibly_overlapping_pairs(
    first: tuple[_Triangle2, ...],
    second: tuple[_Triangle2, ...],
    self_pairs: bool,
    /,
) -> list[tuple[int, int]]:
    """Row pairs (``i < j`` for self pairs) whose exact overlap may be positive.

    Pairs with disjoint closed binary64 boxes, or whose interiors the interval
    screen of the exact edge-line test separates, meet in area zero.
    """
    pairs = box_overlap_pairs(
        len(first),
        len(second),
        polygon_boxes(first),
        polygon_boxes(second),
        self_pairs=self_pairs,
    )
    decisions = triangle_pair_separation(first, second, pairs)
    return [(left, right) for left, right in pairs[decisions != 1].tolist()]


def _volume_face_owners(
    mesh: CellMesh, cell_regions: np.ndarray, /
) -> tuple[
    dict[tuple[int, ...], tuple[tuple[int, int, int], ...]],
    dict[tuple[int, int], tuple[int, ...]],
    dict[tuple[int, int], int],
]:
    from ..discretization._reference_cell import reference_cell_topology

    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    grouped: dict[tuple[int, ...], list[tuple[int, int, int]]] = {}
    by_cell_face: dict[tuple[int, int], tuple[int, ...]] = {}
    face_outward_sign: dict[tuple[int, int], int] = {}
    global_cell = 0
    for block in mesh.blocks:
        reference = reference_cell_topology(block.cell_kind)
        if reference.dimension != 3:
            raise SourceCoverageUnavailable(
                "Source coverage requires volume target cells.", status="unresolved"
            )
        values = np.asarray(block.vertices, dtype=np.int64)
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        reference_vertices: tuple[_Point3, ...] = tuple(
            _point3(tuple(_fraction(float(value)) for value in point))
            for point in reference.vertices
        )
        cell_center: _Point3 = (
            sum((point[0] for point in reference_vertices), Fraction(0))
            / len(reference_vertices),
            sum((point[1] for point in reference_vertices), Fraction(0))
            / len(reference_vertices),
            sum((point[2] for point in reference_vertices), Fraction(0))
            / len(reference_vertices),
        )
        for row in range(block.cell_count):
            cell_id = int(identifiers[row])
            region = int(cell_regions[global_cell])
            for local_face, face in enumerate(reference.entities[2]):
                slots = tuple(int(value) for value in face)
                key = tuple(
                    sorted(int(vertex_ids[int(values[row, slot])]) for slot in slots)
                )
                grouped.setdefault(key, []).append((cell_id, region, local_face))
                by_cell_face[(cell_id, local_face)] = key
                points = tuple(reference_vertices[slot] for slot in slots)
                if len(points) < 3:
                    raise SourceCoverageUnavailable(
                        "Target reference face has fewer than three vertices.",
                        status="unresolved",
                    )
                normal = _cross3((points[0], points[1], points[2]))
                center = (
                    sum((point[0] for point in points), Fraction(0)) / len(points),
                    sum((point[1] for point in points), Fraction(0)) / len(points),
                    sum((point[2] for point in points), Fraction(0)) / len(points),
                )
                outward = _dot3(
                    normal,
                    (
                        center[0] - cell_center[0],
                        center[1] - cell_center[1],
                        center[2] - cell_center[2],
                    ),
                )
                if outward == 0:
                    raise SourceCoverageUnavailable(
                        "Target reference face has no oriented outward normal.",
                        status="unresolved",
                    )
                face_outward_sign[(cell_id, local_face)] = 1 if outward > 0 else -1
            global_cell += 1
    if global_cell != cell_regions.size:
        raise SourceCoverageUnavailable(
            "Target region assignments do not align with volume cells.", status="violated"
        )
    compact: dict[tuple[int, ...], tuple[tuple[int, int, int], ...]] = {}
    for key, owners in grouped.items():
        if len(owners) > 2:
            raise SourceCoverageUnavailable(
                "Target volume has a nonmanifold face.", status="violated"
            )
        compact[key] = tuple(owners)
    return compact, by_cell_face, face_outward_sign


def _cad_pcurve_parameter(
    use: object, point: tuple[Fraction, Fraction], /
) -> Fraction | None:
    from ._meshing_domain import PatchCurveUse, PatchPoleUse
    from .brep._boolean_core import PeriodicPCurve
    from .brep._patches import ExactLineCurve, LineCurve

    if not isinstance(use, (PatchCurveUse, PatchPoleUse)):
        return None
    pcurve = use.pcurve
    if isinstance(pcurve, PeriodicPCurve):
        # A native-period gauge is its source line shifted by its binary64 offset.
        shift = tuple(
            Fraction.from_float(float(value))
            for value in np.asarray(pcurve.offset_value, dtype=np.float64)
        )
        point = (point[0] - shift[0], point[1] - shift[1])
        pcurve = pcurve.source_curve
    if isinstance(pcurve, ExactLineCurve):
        origin = pcurve.first_exact
        direction = tuple(
            last - first
            for first, last in zip(pcurve.first_exact, pcurve.last_exact, strict=True)
        )
    elif isinstance(pcurve, LineCurve):
        origin = pcurve.origin_exact
        direction = pcurve.direction_exact
    else:
        return None
    if len(origin) != 2 or len(direction) != 2:
        return None
    origin_first, origin_second = origin
    direction_first, direction_second = direction
    if (
        not isinstance(origin_first, Fraction)
        or not isinstance(origin_second, Fraction)
        or not isinstance(direction_first, Fraction)
        or not isinstance(direction_second, Fraction)
    ):
        return None
    exact_origin = (origin_first, origin_second)
    exact_direction = (direction_first, direction_second)
    parameter: Fraction | None = None
    for axis in range(2):
        if exact_direction[axis] != 0:
            parameter = (point[axis] - exact_origin[axis]) / exact_direction[axis]
            break
    if parameter is None:
        return None
    lower = min(
        Fraction.from_float(float(use.first)), Fraction.from_float(float(use.last))
    )
    upper = max(
        Fraction.from_float(float(use.first)), Fraction.from_float(float(use.last))
    )
    if not lower <= parameter <= upper:
        return None
    try:
        evaluated = tuple(pcurve.evaluate_exact(parameter))
    except (TypeError, ValueError, ArithmeticError):
        return None
    return parameter if evaluated == point else None


def _cad_chart_vertex_key(
    domain: MeshingDomain,
    patch: int,
    point: _Point2,
    /,
) -> tuple[object, ...]:
    from ._meshing_domain import PatchCurveUse, PatchPoleUse

    labels: set[tuple[object, ...]] = set()
    for loop in domain.patches[patch].loops:
        for use in loop:
            if isinstance(use, PatchPoleUse):
                # A collapsed pole side maps every chart point to its one corner.
                if _cad_pcurve_parameter(use, point) is not None:
                    labels.add(
                        (
                            "corner",
                            domain.entity_id(0, use.corner),
                            domain.source_occurrences[0][use.corner],
                        )
                    )
                continue
            if not isinstance(use, PatchCurveUse):
                continue
            parameter = _cad_pcurve_parameter(use, point)
            if parameter is None:
                continue
            curve_row = int(use.curve)
            curve_id = domain.entity_id(1, curve_row)
            occurrence = domain.source_occurrences[1][curve_row]
            if parameter == Fraction.from_float(float(use.first)):
                corner = domain.curves[curve_row].start
                labels.add(
                    (
                        "corner",
                        domain.entity_id(0, corner),
                        domain.source_occurrences[0][corner],
                    )
                )
            elif parameter == Fraction.from_float(float(use.last)):
                corner = domain.curves[curve_row].end
                labels.add(
                    (
                        "corner",
                        domain.entity_id(0, corner),
                        domain.source_occurrences[0][corner],
                    )
                )
            else:
                labels.add(("curve", curve_id, occurrence, parameter))
    if len(labels) > 1:
        raise SourceCoverageUnavailable(
            "A CAD chart vertex lies on conflicting authored curve occurrences.",
            status="unresolved",
        )
    if labels:
        return next(iter(labels))
    return (
        "patch",
        domain.entity_id(2, patch),
        domain.source_occurrences[2][patch],
        point,
    )


def _cad_chart_curve_use(
    domain: MeshingDomain,
    patch: int,
    first: _Point2,
    second: _Point2,
    /,
) -> tuple[tuple[str, tuple[str, ...]], Fraction, Fraction, PatchCurveUse]:
    from ._meshing_domain import PatchCurveUse

    matches: list[
        tuple[tuple[str, tuple[str, ...]], Fraction, Fraction, PatchCurveUse]
    ] = []
    for loop in domain.patches[patch].loops:
        for use in loop:
            if not isinstance(use, PatchCurveUse):
                continue
            first_parameter = _cad_pcurve_parameter(use, first)
            second_parameter = _cad_pcurve_parameter(use, second)
            if (
                first_parameter is None
                or second_parameter is None
                or first_parameter == second_parameter
            ):
                continue
            curve_row = int(use.curve)
            key = (
                domain.entity_id(1, curve_row),
                domain.source_occurrences[1][curve_row],
            )
            matches.append((key, first_parameter, second_parameter, use))
    if len(matches) != 1:
        raise SourceCoverageUnavailable(
            "A CAD chart seam does not select one authored affine coedge occurrence.",
            status="unresolved",
        )
    return matches[0]


def _cad_trim_root_box(junction: _RootedJunction, /) -> np.ndarray:
    from ._rational import fraction_down, fraction_up

    return np.asarray(
        (
            (fraction_down(junction.lower[0]), fraction_down(junction.lower[1])),
            (fraction_up(junction.upper[0]), fraction_up(junction.upper[1])),
        ),
        dtype=np.float64,
    )


def _cad_trim_reference_loops(
    charts: SurfaceSourceCharts,
    bindings: dict[int, tuple[_CadCurveUseBinding, ...]],
    patch: int,
    /,
) -> tuple[tuple[TrimReferenceSegment, ...], ...]:
    """Resolve each authored coedge to its certified root-chart reference chain."""
    from ._domain_boundary import (
        _rooted_patch_chart_endpoints,
        _rooted_patch_chart_junctions,
    )
    from ._meshing_domain import PatchCurveUse, PatchPoleUse, SourcePreparation
    from ._surface_source_support import _cad_root_edges
    from .brep._curve_approximation_topology import TrimReferenceSegment

    domain = charts.domain
    authority = domain.brep_authority
    if authority is None:
        raise SourceCoverageUnavailable(
            "CAD trim homotopy requires BRep authority.", status="unresolved"
        )
    face = domain.source_indices[2][patch]
    preparation = SourcePreparation(domain)
    junctions = _rooted_patch_chart_junctions(domain, patch, preparation)
    endpoints = _rooted_patch_chart_endpoints(domain, patch)
    _, adjacency = _cad_root_edges(charts)
    patch_by_cell = {
        int(identifier): int(charts.root_patches[row])
        for row, identifier in enumerate(charts.cell_global_ids)
    }
    by_use: dict[tuple[int, int], list[tuple[_CadCurveUseBinding, _Point2, _Point2]]] = {}
    for edge_id, records in bindings.items():
        for record in records:
            if patch_by_cell[record.root_cell_id] != patch:
                continue
            incident = tuple(
                item for item in adjacency[edge_id] if item[0] == record.root_cell_id
            )
            if len(incident) != 1:
                raise SourceCoverageUnavailable(
                    "CAD coedge lost its unique root-chart edge.", status="violated"
                )
            by_use.setdefault((record.loop_index, record.use_index), []).append(
                (record, incident[0][2], incident[0][3])
            )
    result: list[tuple[TrimReferenceSegment, ...]] = []
    for loop_index, loop in enumerate(domain.patches[patch].loops):
        output: list[TrimReferenceSegment] = []
        coedges = authority.geometry.face_loops[face][loop_index]
        for use_index, use in enumerate(loop):
            previous = junctions[loop_index][use_index - 1]
            following = junctions[loop_index][use_index]

            start_root = authority.geometry.coedge_endpoint_roots[coedges[use_index]][
                0 if use.sense > 0 else 1
            ]
            stop_root = authority.geometry.coedge_endpoint_roots[coedges[use_index]][
                1 if use.sense > 0 else 0
            ]
            start_parameter_box = (
                None if start_root is None else start_root.parameter_enclosure()
            )
            stop_parameter_box = (
                None if stop_root is None else stop_root.parameter_enclosure()
            )
            if isinstance(use, PatchPoleUse):
                start, stop = endpoints[loop_index][use_index]
                first, last = (
                    (Fraction.from_float(use.first), Fraction.from_float(use.last))
                    if use.sense > 0
                    else (Fraction.from_float(use.last), Fraction.from_float(use.first))
                )
                output.append(
                    TrimReferenceSegment(
                        use.pcurve,
                        first,
                        last,
                        start,
                        stop,
                        _cad_trim_root_box(previous),
                        _cad_trim_root_box(following),
                        start_parameter_box,
                        stop_parameter_box,
                    )
                )
                continue
            if not isinstance(use, PatchCurveUse):
                raise SourceCoverageUnavailable(
                    "CAD trim has an unsupported coedge.", status="unresolved"
                )
            records = by_use.get((loop_index, use_index), ())
            ordered = sorted(
                records, key=lambda value: min(value[0].parameters), reverse=use.sense < 0
            )
            if not ordered:
                raise SourceCoverageUnavailable(
                    "CAD source coedge has no complete root edge chain.",
                    status="violated",
                )
            for record, canonical_first, canonical_last in ordered:
                p0, p1 = record.parameters
                if (p1 > p0) != (use.sense > 0):
                    p0, p1 = p1, p0
                    canonical_first, canonical_last = canonical_last, canonical_first
                base = Fraction.from_float(use.first)
                span = Fraction.from_float(use.last) - base
                is_start = p0 == (0 if use.sense > 0 else 1)
                is_stop = p1 == (1 if use.sense > 0 else 0)
                output.append(
                    TrimReferenceSegment(
                        use.pcurve,
                        base + span * p0,
                        base + span * p1,
                        canonical_first,
                        canonical_last,
                        _cad_trim_root_box(previous) if is_start else None,
                        _cad_trim_root_box(following) if is_stop else None,
                        start_parameter_box if is_start else None,
                        stop_parameter_box if is_stop else None,
                    )
                )
        result.append(tuple(output))
    return tuple(result)


def _cad_corner_key(domain: MeshingDomain, corner: int, /) -> tuple[object, ...]:
    return ("corner", domain.entity_id(0, corner), domain.source_occurrences[0][corner])


def _cad_plane_point_keys(
    domain: MeshingDomain,
    patch: int,
    curve_keys: dict[int, set[tuple[object, ...]]],
    /,
) -> dict[_Point2, tuple[object, ...]]:
    """Authored keys of the plane chart points of a patch's trim samples.

    A trim sample keyed by another patch as an authored line p-curve parameter
    is located on this patch's own authored use of that curve, then mapped by
    the exact plane chart; trim ends map to their corners and the pole to its
    corner. Plane chart points not reached this way are patch-interior points.
    """
    from ._domain_boundary import exact_plane_chart
    from ._meshing_domain import PatchCurveUse, PatchPoleUse
    from .brep._patches import ExactLineCurve, LineCurve

    plane = exact_plane_chart(domain, patch)
    if plane is None:
        raise SourceCoverageUnavailable(
            "A CAD plane chart row names a patch without an exact plane chart.",
            status="violated",
        )
    keys: dict[_Point2, tuple[object, ...]] = {}

    def bind(point: _Point2, key: tuple[object, ...], /) -> None:
        if keys.setdefault(point, key) != key:
            raise SourceCoverageUnavailable(
                "Distinct authored CAD points meet in one plane chart point.",
                status="unresolved",
            )

    for loop in domain.patches[patch].loops:
        for use in loop:
            if isinstance(use, PatchPoleUse):
                bind((Fraction(0), Fraction(0)), _cad_corner_key(domain, use.corner))
                continue
            if not isinstance(use, PatchCurveUse) or not isinstance(
                use.pcurve, (ExactLineCurve, LineCurve)
            ):
                continue
            curve = domain.curves[use.curve]
            for parameter, corner in ((use.first, curve.start), (use.last, curve.end)):
                point = tuple(use.pcurve.evaluate_exact(Fraction.from_float(parameter)))
                bind(
                    plane.chart_point(_triangle_point(point)),
                    _cad_corner_key(domain, corner),
                )
            for key in curve_keys.get(int(use.curve), ()):
                parameter = key[3]
                if not isinstance(parameter, Fraction):
                    continue
                point = tuple(use.pcurve.evaluate_exact(parameter))
                bind(plane.chart_point(_triangle_point(point)), key)
    return keys


def _triangle_point(values: tuple[object, ...], /) -> _Point2:
    first, second = values
    if not isinstance(first, Fraction) or not isinstance(second, Fraction):
        raise SourceCoverageUnavailable(
            "A CAD trim p-curve point is not rational.", status="unresolved"
        )
    return first, second


def _cad_plane_curve_use(
    domain: MeshingDomain,
    patch: int,
    first: tuple[object, ...],
    second: tuple[object, ...],
    /,
) -> tuple[tuple[str, tuple[str, ...]], Fraction, Fraction, PatchCurveUse]:
    """Authored coedge of a plane-charted patch between two keyed trim points.

    Each end takes its key's p-curve parameter on the patch's own authored use;
    a corner end of a closed curve takes the use end nearer the other end.
    """
    from ._meshing_domain import PatchCurveUse

    matches: list[
        tuple[tuple[str, tuple[str, ...]], Fraction, Fraction, PatchCurveUse]
    ] = []
    for loop in domain.patches[patch].loops:
        for use in loop:
            if not isinstance(use, PatchCurveUse):
                continue
            curve_row = int(use.curve)
            curve_key = (
                domain.entity_id(1, curve_row),
                domain.source_occurrences[1][curve_row],
            )
            curve = domain.curves[curve_row]
            ends = (Fraction.from_float(use.first), Fraction.from_float(use.last))
            parameters: list[Fraction | None] = []
            for key in (first, second):
                if key[0] == "curve" and (key[1], key[2]) == curve_key:
                    parameter = key[3]
                    parameters.append(
                        parameter if isinstance(parameter, Fraction) else None
                    )
                elif key[0] == "corner" and key in (
                    _cad_corner_key(domain, curve.start),
                    _cad_corner_key(domain, curve.end),
                ):
                    candidates = [
                        end
                        for end, corner in zip(
                            ends, (curve.start, curve.end), strict=True
                        )
                        if _cad_corner_key(domain, corner) == key
                    ]
                    parameters.append(candidates[0] if len(candidates) == 1 else None)
                else:
                    break
            else:
                first_parameter, second_parameter = parameters
                if first_parameter is None and second_parameter is not None:
                    known = second_parameter
                    first_parameter = min(ends, key=lambda end: abs(end - known))
                if second_parameter is None and first_parameter is not None:
                    known = first_parameter
                    second_parameter = min(ends, key=lambda end: abs(end - known))
                if (
                    first_parameter is not None
                    and second_parameter is not None
                    and first_parameter != second_parameter
                ):
                    matches.append((curve_key, first_parameter, second_parameter, use))
    if len(matches) != 1:
        raise SourceCoverageUnavailable(
            "A CAD chart seam does not select one authored affine coedge occurrence.",
            status="unresolved",
        )
    return matches[0]


@dataclass(frozen=True, slots=True)
class _CadTargetAncestry:
    vertex_keys: dict[int, _VertexKey]
    edge_uses: dict[
        tuple[int, tuple[int, int]],
        tuple[tuple[str, tuple[str, ...]], Fraction, Fraction, PatchCurveUse],
    ]


def _cad_target_root_partition(
    domain: MeshingDomain,
    mesh: CellMesh,
    witness: SourceBoundaryChartWitness,
    patch_rows: list[int],
    tolerance: float,
    /,
) -> _CadTargetAncestry:
    """Replay exact target restrictions of the retained CAD root charts."""
    from ..discretization._reference_cell import reference_cell_topology
    from ..meshing._association import (
        GeometryAssociationArchiveRecipe,
        GeometryAssociationKind,
        GeometryAssociationProvenance,
        GeometrySourceEntityRole,
    )
    from ..meshing._plc_source import PiecewiseLinearComplex
    from ..meshing.providers._native_cad_volume import _exact_patch_chart
    from ..meshing.providers._native_plc_geometry import simplex_volume_entities
    from ._domain_boundary import exact_plane_chart
    from ._meshing_domain import PatchCurveUse, SourcePreparation
    from ._surface_source_support import (
        _cad_plane_seam_side,
        _cad_pole_lens_endpoint,
        _cad_root_edges,
        validate_cad_curve_ancestry,
    )

    if witness.cad_plc is None:
        raise SourceCoverageUnavailable(
            "CAD root ancestry is absent.", status="unresolved"
        )
    try:
        plc = PiecewiseLinearComplex.from_stored_fields(witness.cad_plc)
        charts = plc.cad_source_charts
        if charts is None or not _same_domain(charts.domain, domain):
            raise ValueError("CAD root charts bind another source.")
        charts.require_current()
        bindings = validate_cad_curve_ancestry(charts, tolerance)
        _, root_edges = _cad_root_edges(charts)
    except (TypeError, ValueError) as error:
        raise SourceCoverageUnavailable(
            f"CAD root ancestry is invalid: {error}", status="violated"
        ) from error

    root_rows = {identifier: row for row, identifier in enumerate(charts.cell_global_ids)}
    roots = np.asarray(witness.cad_root_cell_ids, dtype=np.int64)
    if roots.shape != (len(patch_rows),):
        raise SourceCoverageUnavailable(
            "CAD root rows are misaligned.", status="violated"
        )
    cell_rows = {
        int(identifier): (block, row)
        for block in mesh.blocks
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    targets: list[tuple[int, int, int]] = []
    triangles: list[_Triangle2] = []
    grouped: dict[int, list[_Triangle2]] = {}
    plane_patches: set[int] = set()
    planes = {patch: exact_plane_chart(domain, patch) for patch in set(patch_rows)}
    affine_patches = {
        patch
        for patch in {
            patch for root, patch in zip(roots, patch_rows, strict=True) if root == -1
        }
        if _exact_patch_chart(domain, patch) is not None
    }
    for row, (root_id, patch) in enumerate(zip(roots, patch_rows, strict=True)):
        triangle = _triangle2(witness.source_parameter_chart(row))
        if _area2(triangle) == 0:
            raise SourceCoverageUnavailable(
                "CAD target root chart is degenerate.", status="violated"
            )
        triangles.append(triangle)
        found = cell_rows.get(int(witness.cell_global_ids[row]))
        if found is None:
            raise SourceCoverageUnavailable(
                "CAD target cell is absent.", status="violated"
            )
        block, cell_row = found
        reference = reference_cell_topology(block.cell_kind)
        local_face = int(witness.local_entity_indices[row])
        if not 0 <= local_face < len(reference.entities[2]):
            raise SourceCoverageUnavailable(
                "CAD target face is invalid.", status="violated"
            )
        slots = set(reference.entities[2][local_face])
        corners = tuple(
            tuple(_fraction(float(x)) for x in point) for point in reference.vertices
        )
        ids: list[int] = []
        for point in _triangle3(witness.target_reference_chart(row)):
            matches = [
                slot
                for slot, corner in enumerate(corners)
                if point == corner and slot in slots
            ]
            if len(matches) != 1:
                raise SourceCoverageUnavailable(
                    "CAD target corner is not a face vertex.", status="violated"
                )
            ids.append(
                int(vertex_ids[int(np.asarray(block.vertices)[cell_row, matches[0]])])
            )
        if len(set(ids)) != 3:
            raise SourceCoverageUnavailable(
                "CAD target face repeats a vertex.", status="violated"
            )
        targets.append((ids[0], ids[1], ids[2]))
        if (
            witness.source_entity_ids[row] != domain.entity_id(2, patch)
            or witness.occurrence_paths[row] != domain.source_occurrences[2][patch]
        ):
            raise SourceCoverageUnavailable(
                "CAD target row changes source identity.", status="violated"
            )
        plane = planes[patch]
        if (
            plane is not None
            and witness.source_chart_ids[row] == plane.chart_id
            and root_id != -1
        ):
            raise SourceCoverageUnavailable(
                "A derived CAD plane chart cannot retain a nonlinear root triangle.",
                status="violated",
            )
        if root_id >= 0:
            root_row = root_rows.get(int(root_id))
            if root_row is None or int(charts.root_patches[root_row]) != patch:
                raise SourceCoverageUnavailable(
                    "CAD target row changes root patch.", status="violated"
                )
            root = _triangle2(charts.root_parameter_chart(root_row))
            if _area2(root) == 0 or _intersection_area(triangle, root) != abs(
                _area2(triangle)
            ):
                raise SourceCoverageUnavailable(
                    "CAD target row leaves its root chart.", status="violated"
                )
        if plane is not None and witness.source_chart_ids[row] == plane.chart_id:
            plane_patches.add(patch)
            continue
        if patch in affine_patches:
            if witness.source_chart_ids[row] != domain.entity_id(2, patch):
                raise SourceCoverageUnavailable(
                    "CAD affine plane changes source chart.", status="violated"
                )
            continue
        if root_id == -1:
            raise SourceCoverageUnavailable(
                "Unrooted CAD chart is not an exact plane.", status="violated"
            )
        if witness.source_chart_ids[row] != domain.entity_id(2, patch):
            raise SourceCoverageUnavailable(
                "CAD target row changes root source identity.", status="violated"
            )
        grouped.setdefault(int(root_id), []).append(triangle)
    rows_by_target_edge: dict[tuple[int, int], list[int]] = {}
    for row, vertices in enumerate(targets):
        for first, second in (
            (vertices[0], vertices[1]),
            (vertices[1], vertices[2]),
            (vertices[2], vertices[0]),
        ):
            key = (min(first, second), max(first, second))
            rows_by_target_edge.setdefault(key, []).append(row)
    for root_id, root_row in root_rows.items():
        patch = int(charts.root_patches[root_row])
        if patch in plane_patches or patch in affine_patches:
            if root_id in grouped:
                raise SourceCoverageUnavailable(
                    "CAD plane mixes source charts.", status="violated"
                )
            continue
        root = _triangle2(charts.root_parameter_chart(root_row))
        rows = grouped.get(root_id, [])
        if not rows or sum((abs(_area2(t)) for t in rows), Fraction(0)) != abs(
            _area2(root)
        ):
            raise SourceCoverageUnavailable(
                "CAD target rows do not cover their root chart.", status="violated"
            )
        for first, second in combinations(rows, 2):
            if _intersection_area(first, second):
                raise SourceCoverageUnavailable(
                    "CAD target root rows overlap.", status="violated"
                )

    for patch in plane_patches | affine_patches:
        rows = tuple(
            triangles[row] for row, value in enumerate(patch_rows) if value == patch
        )
        plane = planes[patch]
        if patch in plane_patches:
            if plane is None or any(
                witness.source_chart_ids[row] != plane.chart_id
                for row, value in enumerate(patch_rows)
                if value == patch
            ):
                raise SourceCoverageUnavailable(
                    "CAD plane mixes source charts.", status="violated"
                )
            declared = charts.boundary_source._trim_partition_reference_unchecked(
                patch, rows, SourcePreparation(domain), plane=True
            )[0]
        else:
            declared = dict(charts.boundary_source.boundary_parameter_triangles())[patch]
        source_area = sum((abs(_area2(t)) for t in declared), Fraction(0))
        if (
            not declared
            or sum((abs(_area2(t)) for t in rows), Fraction(0)) != source_area
            or any(
                sum((_intersection_area(t, root) for root in declared), Fraction(0))
                != abs(_area2(t))
                for t in rows
            )
            or any(
                _intersection_area(rows[first], rows[second])
                for first, second in _possibly_overlapping_pairs(rows, rows, True)
            )
        ):
            raise SourceCoverageUnavailable(
                "CAD plane rows do not partition the exact plane reference.",
                status="violated",
            )

    curve_associations = [
        value
        for value in charts.associations
        if value.association_kind is GeometryAssociationKind.CURVE
        and value.target_entity_set_id == charts.mesh.entity_set(1).entity_set_id
    ]
    vertex_keys: dict[int, _VertexKey] = {}
    edge_uses: dict[
        tuple[int, tuple[int, int]],
        tuple[tuple[str, tuple[str, ...]], Fraction, Fraction, PatchCurveUse],
    ] = {}
    association = witness.cad_curve_association
    if bindings:
        if len(curve_associations) != 1 or association is None:
            raise SourceCoverageUnavailable(
                "CAD target curve lineage is absent.", status="violated"
            )
        original = GeometryAssociationArchiveRecipe.from_association(
            curve_associations[0]
        ).restore()
        association = GeometryAssociationArchiveRecipe.from_association(
            association
        ).restore()
        association.validate_target(mesh.entity_set(1))
        if (
            association.provenance is not GeometryAssociationProvenance.LINEAGE
            or association.parent_association_id != original.association_id
            or association.source_id != charts.source_id
            or association.source_revision != charts.source_revision
        ):
            raise SourceCoverageUnavailable(
                "CAD target curve lineage changes its parent.", status="violated"
            )
        originals = {
            int(value): row
            for row, value in enumerate(np.asarray(original.target_global_ids))
        }
        edge_vertices, _ = simplex_volume_entities(mesh)
        edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
        target_edges = {
            int(identifier): tuple(int(vertex_ids[int(v)]) for v in edge)
            for identifier, edge in zip(edge_ids, edge_vertices, strict=True)
        }
        parts: dict[int, list[tuple[Fraction, Fraction]]] = {}
        for index, raw_edge in enumerate(np.asarray(association.target_global_ids)):
            edge_id = int(raw_edge)
            parent = int(association.parent_ids[index])
            old = originals.get(parent)
            interval = association.source_parameter_intervals[index]
            if (
                old is None
                or parent not in bindings
                or edge_id not in target_edges
                or int(association.parent_dimensions[index]) != 1
                or int(association.source_dimensions[index]) != 1
                or association.source_roles[index] is not GeometrySourceEntityRole.EDGE
                or not bool(association.resolved[index])
                or bool(association.ambiguous[index])
                or association.source_entity_ids[index] != original.source_entity_ids[old]
                or int(association.source_indices[index])
                != int(original.source_indices[old])
                or association.source_occurrence_paths[index]
                != original.source_occurrence_paths[old]
                or interval is None
                or original.source_parameter_intervals[old] is None
            ):
                raise SourceCoverageUnavailable(
                    "CAD target curve changes root ancestry.", status="violated"
                )
            root_interval = original.source_parameter_intervals[old]
            assert root_interval is not None
            if not root_interval[0] <= interval[0] < interval[1] <= root_interval[1]:
                raise SourceCoverageUnavailable(
                    "CAD target curve interval leaves its root.", status="violated"
                )
            parts.setdefault(parent, []).append(interval)
            binding0 = bindings[parent][0]
            curve = binding0.curve_row
            owner_patch, owner_loop, owner_index = domain.curve_owners[curve]
            owner = domain.patches[owner_patch].loops[owner_loop][owner_index]
            sense = 1 if owner.last > owner.first else -1
            direction = int(association.orientations[index]) * sense
            if direction not in (-1, 1):
                raise SourceCoverageUnavailable(
                    "CAD target curve orientation is invalid.", status="violated"
                )
            endpoint_parameters = (
                interval if direction > 0 else (interval[1], interval[0])
            )
            endpoints = target_edges[edge_id]
            for vertex, parameter in zip(endpoints, endpoint_parameters, strict=True):
                if parameter in (0, 1):
                    corner = (
                        domain.curves[curve].start
                        if parameter == 0
                        else domain.curves[curve].end
                    )
                    key = _cad_corner_key(domain, corner)
                else:
                    actual = Fraction.from_float(float(owner.first)) + parameter * (
                        Fraction.from_float(float(owner.last))
                        - Fraction.from_float(float(owner.first))
                    )
                    key = (
                        "curve",
                        domain.entity_id(1, curve),
                        domain.source_occurrences[1][curve],
                        actual,
                    )
                previous = vertex_keys.setdefault(vertex, key)
                if previous != key:
                    raise SourceCoverageUnavailable(
                        "CAD target vertex changes source occurrence.", status="violated"
                    )
            sorted_edge = (min(endpoints), max(endpoints))
            incident_rows = rows_by_target_edge.get(sorted_edge, ())
            for row in incident_rows:
                row_targets = targets[row]
                matches = [
                    binding
                    for binding in bindings[parent]
                    if (
                        int(charts.root_patches[root_rows[binding.root_cell_id]])
                        == patch_rows[row]
                        and (
                            binding.root_cell_id == int(roots[row])
                            or (
                                roots[row] == -1
                                and (
                                    patch_rows[row] in affine_patches
                                    or patch_rows[row] in plane_patches
                                )
                            )
                        )
                    )
                ]
                if len(matches) > 1 and patch_rows[row] in plane_patches:
                    plane = planes[patch_rows[row]]
                    target_slots = tuple(row_targets.index(vertex) for vertex in endpoints)
                    sided = []
                    for candidate in matches:
                        incidence = [
                            item
                            for item in root_edges[parent]
                            if item[0] == candidate.root_cell_id
                        ]
                        if len(incidence) != 1:
                            continue
                        _, slot, first_uv, last_uv = incidence[0]
                        first_uv = _cad_pole_lens_endpoint(
                            charts, patch_rows[row], candidate.loop_index,
                            candidate.use_index, first_uv, candidate.parameters[0],
                        )
                        last_uv = _cad_pole_lens_endpoint(
                            charts, patch_rows[row], candidate.loop_index,
                            candidate.use_index, last_uv, candidate.parameters[1],
                        )
                        root = charts.root_parameter_chart(root_rows[candidate.root_cell_id])
                        if _cad_plane_seam_side(
                            plane,
                            root,
                            slot,
                            (first_uv, last_uv),
                            triangles[row],
                            target_slots,
                            candidate.parameters,
                            endpoint_parameters,
                        ):
                            sided.append(candidate)
                    matches = sided
                if len(matches) != 1:
                    raise SourceCoverageUnavailable(
                        f"CAD target edge {edge_id} (root {parent}, patch {patch_rows[row]}, "
                        f"chart {roots[row]}) lacks its root coedge: "
                        f"{len(matches)} matching occurrences.",
                        status="violated",
                    )
                binding = matches[0]
                incidence = [
                    item for item in root_edges[parent] if item[0] == binding.root_cell_id
                ]
                if len(incidence) != 1:
                    raise SourceCoverageUnavailable(
                        "CAD root edge incidence is ambiguous.", status="violated"
                    )
                first_uv, last_uv = incidence[0][2:]
                if patch_rows[row] in plane_patches:
                    plane = exact_plane_chart(domain, patch_rows[row])
                    if plane is None:
                        raise SourceCoverageUnavailable(
                            "CAD plane chart authority is missing.", status="violated"
                        )
                    first_uv, last_uv = (
                        plane.chart_point(first_uv),
                        plane.chart_point(last_uv),
                    )
                start, stop = binding.parameters
                row_uv = dict(zip(row_targets, triangles[row], strict=True))
                for vertex, parameter in zip(endpoints, endpoint_parameters, strict=True):
                    t = (parameter - start) / (stop - start)
                    expected = tuple(
                        a + t * (b - a) for a, b in zip(first_uv, last_uv, strict=True)
                    )
                    if row_uv[vertex] != expected:
                        raise SourceCoverageUnavailable(
                            "CAD target endpoint changes its root edge chart.",
                            status="violated",
                        )
                use = domain.patches[patch_rows[row]].loops[binding.loop_index][
                    binding.use_index
                ]
                if not isinstance(use, PatchCurveUse):
                    raise SourceCoverageUnavailable(
                        "CAD curve ancestry names a collapsed pole use.",
                        status="violated",
                    )
                source_parameters = tuple(
                    Fraction.from_float(float(use.first))
                    + p
                    * (
                        Fraction.from_float(float(use.last))
                        - Fraction.from_float(float(use.first))
                    )
                    for p in endpoint_parameters
                )
                ordered = (
                    source_parameters
                    if endpoints == sorted_edge
                    else source_parameters[::-1]
                )
                edge_uses[(row, sorted_edge)] = (
                    (domain.entity_id(1, curve), domain.source_occurrences[1][curve]),
                    ordered[0],
                    ordered[1],
                    use,
                )
            if not incident_rows:
                raise SourceCoverageUnavailable(
                    "CAD target curve edge has no boundary chart incidence.",
                    status="violated",
                )
        for parent in bindings:
            old = originals[parent]
            root_interval = original.source_parameter_intervals[old]
            children = sorted(parts.get(parent, ()))
            if (
                root_interval is None
                or not children
                or children[0][0] != root_interval[0]
                or children[-1][1] != root_interval[1]
                or any(a[1] != b[0] for a, b in zip(children, children[1:]))
            ):
                raise SourceCoverageUnavailable(
                    "CAD target curve intervals do not partition their root.",
                    status="violated",
                )
    elif association is not None:
        raise SourceCoverageUnavailable(
            "CAD target curve lineage has no root curves.", status="violated"
        )
    for patch, target_ids, triangle in zip(patch_rows, targets, triangles, strict=True):
        for vertex, point in zip(target_ids, triangle, strict=True):
            key = vertex_keys.get(vertex)
            if key is not None and key[0] != "patch":
                continue
            interior = (
                "patch",
                domain.entity_id(2, patch),
                domain.source_occurrences[2][patch],
                point,
            )
            if key is not None and key != interior:
                raise SourceCoverageUnavailable(
                    "CAD interior vertex changes source chart.", status="violated"
                )
            vertex_keys[vertex] = interior
    return _CadTargetAncestry(vertex_keys, edge_uses)


def _cad_validate_coedge_pair(
    first_use: tuple[tuple[str, tuple[str, ...]], Fraction, Fraction, PatchCurveUse]
    | None,
    second_use: tuple[tuple[str, tuple[str, ...]], Fraction, Fraction, PatchCurveUse]
    | None,
    first_region_sign: int,
    second_region_sign: int,
    /,
) -> None:
    if first_use is None or second_use is None:
        raise SourceCoverageUnavailable(
            "CAD seam lacks bound root coedges.", status="violated"
        )
    if first_use[:3] != second_use[:3]:
        raise SourceCoverageUnavailable(
            "Adjacent CAD charts do not map their shared target edge to the same authored coedge occurrence and traversal.",
            status="violated",
        )
    if (
        first_region_sign * first_use[3].sense + second_region_sign * second_use[3].sense
        != 0
    ):
        raise SourceCoverageUnavailable(
            "Adjacent CAD chart coedges do not have opposite material-region traversal.",
            status="violated",
        )


def _cad_chart_topology_degree(
    domain: MeshingDomain,
    mesh: CellMesh,
    witness: SourceBoundaryChartWitness,
    patch_rows: list[int],
    pairs: tuple[tuple[int, int], ...],
    signs_by_patch: tuple[tuple[tuple[int, int], ...], ...],
    by_cell_face: dict[tuple[int, int], tuple[int, ...]],
    cell_region: dict[int, int],
    face_outward_sign: dict[tuple[int, int], int],
    plane_patches: frozenset[int],
    /,
    ancestry: _CadTargetAncestry | None = None,
) -> tuple[tuple[tuple[int, ...], int, int, int, int, int], ...]:
    """Prove a per-region chart homeomorphism and compute its oriented degree.

    Rows of ``plane_patches`` are charted in their patch's exact plane chart;
    their trim points take the authored keys their neighbors give them.
    """
    from ..discretization._reference_cell import reference_cell_topology

    cell_rows = {
        int(identifier): (block, row)
        for block in mesh.blocks
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    row_targets: list[tuple[int, int, int]] = []
    row_uv: list[_Triangle2] = []
    row_labels: list[tuple[_VertexKey, _VertexKey, _VertexKey]] = []
    for row, (cell_raw, local_raw, patch) in enumerate(
        zip(
            witness.cell_global_ids, witness.local_entity_indices, patch_rows, strict=True
        )
    ):
        cell_id, local_face = int(cell_raw), int(local_raw)
        found = cell_rows.get(cell_id)
        face_key = by_cell_face.get((cell_id, local_face))
        if found is None or face_key is None:
            raise SourceCoverageUnavailable(
                "A CAD chart triangle names an absent target face.", status="violated"
            )
        block, cell_row = found
        reference = reference_cell_topology(block.cell_kind)
        faces = reference.entities[2]
        if local_face < 0 or local_face >= len(faces):
            raise SourceCoverageUnavailable(
                "A CAD chart triangle names an invalid target face.", status="violated"
            )
        face_slots = {int(value) for value in faces[local_face]}
        reference_vertices = tuple(
            tuple(_fraction(float(value)) for value in point)
            for point in reference.vertices
        )
        ids = []
        target_chart = _triangle3(witness.target_reference_chart(row))
        for exact_point in target_chart:
            matches = tuple(
                slot
                for slot, corner in enumerate(reference_vertices)
                if corner == exact_point
            )
            if len(matches) != 1 or matches[0] not in face_slots:
                raise SourceCoverageUnavailable(
                    "CAD seam proof currently requires target chart corners at stable volume-mesh vertices.",
                    status="unresolved",
                )
            ids.append(
                int(
                    vertex_ids[
                        int(
                            np.asarray(block.vertices, dtype=np.int64)[
                                cell_row, matches[0]
                            ]
                        )
                    ]
                )
            )
        if len(ids) != 3 or len(set(ids)) != 3 or not set(ids) <= set(face_key):
            raise SourceCoverageUnavailable(
                "A CAD chart triangle changes its target face vertex ancestry.",
                status="violated",
            )
        uv = _triangle2(witness.source_parameter_chart(row))
        if _area2(uv) == 0:
            raise SourceCoverageUnavailable(
                "A CAD source chart triangle is degenerate.", status="violated"
            )
        row_targets.append((ids[0], ids[1], ids[2]))
        row_uv.append(uv)
        # Plane-chart rows are labelled below, once every curve label is known.
        if patch in plane_patches:
            row_labels.append(((), (), ()))
        else:
            labels = tuple(
                ancestry.vertex_keys.get(
                    vertex,
                    (
                        "patch",
                        domain.entity_id(2, patch),
                        domain.source_occurrences[2][patch],
                        point,
                    ),
                )
                if ancestry is not None
                else _cad_chart_vertex_key(domain, patch, point)
                for vertex, point in zip(ids, uv, strict=True)
            )
            row_labels.append((labels[0], labels[1], labels[2]))
    if plane_patches:
        curve_rows: dict[tuple[object, object], int] = {
            (domain.entity_id(1, curve), domain.source_occurrences[1][curve]): curve
            for curve in range(len(domain.curves))
        }
        curve_keys: dict[int, set[tuple[object, ...]]] = {}
        for labels, patch in zip(row_labels, patch_rows, strict=True):
            if patch in plane_patches:
                continue
            for label in labels:
                if label[0] == "curve":
                    curve_keys.setdefault(curve_rows[(label[1], label[2])], set()).add(
                        label
                    )
        plane_keys = {
            patch: _cad_plane_point_keys(domain, patch, curve_keys)
            for patch in plane_patches
        }
        for row, patch in enumerate(patch_rows):
            if patch not in plane_patches:
                continue
            keys = plane_keys[patch]
            first, second, third = (
                keys.get(
                    point,
                    (
                        "patch",
                        domain.entity_id(2, patch),
                        domain.source_occurrences[2][patch],
                        point,
                    ),
                )
                for point in row_uv[row]
            )
            row_labels[row] = (first, second, third)
            if ancestry is not None:
                labels = tuple(
                    ancestry.vertex_keys.get(vertex, label)
                    for vertex, label in zip(
                        row_targets[row], row_labels[row], strict=True
                    )
                )
                row_labels[row] = (labels[0], labels[1], labels[2])

    target_edges: dict[tuple[int, tuple[int, int]], list[int]] = {}
    source_edges: dict[
        tuple[int, tuple[_VertexKey, _VertexKey]], list[tuple[int, int]]
    ] = {}
    source_edge_for_target: dict[
        tuple[int, tuple[int, int]], tuple[_VertexKey, _VertexKey]
    ] = {}
    target_to_source: dict[int, dict[int, _VertexKey]] = {}
    source_to_target: dict[int, dict[_VertexKey, int]] = {}
    row_degree_signs: dict[int, dict[int, int]] = {}
    for row, patch in enumerate(patch_rows):
        pair = pairs[patch]
        uv = row_uv[row]
        targets = row_targets[row]
        labels = row_labels[row]
        chart_sign = 1 if _area2(uv) > 0 else -1
        patch_signs = dict(signs_by_patch[patch])
        cell_id = int(witness.cell_global_ids[row])
        local_face = int(witness.local_entity_indices[row])
        parent_region = cell_region.get(cell_id)
        target_outward = face_outward_sign.get((cell_id, local_face), 0)
        if parent_region is None or target_outward not in (-1, 1):
            raise SourceCoverageUnavailable(
                "CAD chart row has no bound target material-side orientation.",
                status="violated",
            )
        for region in (value for value in pair if value >= 0):
            region_sign = patch_signs.get(region, 0)
            if region_sign not in (-1, 1):
                raise SourceCoverageUnavailable(
                    "CAD source chart omits its material-side orientation.",
                    status="violated",
                )
            target_side_sign = (
                1 if parent_region == region else -1 if parent_region in pair else 0
            )
            if target_side_sign == 0:
                raise SourceCoverageUnavailable(
                    "CAD chart row does not bind one of its target material regions.",
                    status="violated",
                )
            local_degree = (
                int(witness.orientations[row])
                * target_outward
                * target_side_sign
                * region_sign
            )
            if local_degree != 1:
                raise SourceCoverageUnavailable(
                    "CAD source-to-target chart map has nonpositive material degree.",
                    status="violated",
                )
            row_degree_signs.setdefault(region, {})[row] = local_degree
            by_target = target_to_source.setdefault(region, {})
            by_source = source_to_target.setdefault(region, {})
            for target_id, source_key in zip(targets, labels, strict=True):
                previous_source = by_target.setdefault(target_id, source_key)
                previous_target = by_source.setdefault(source_key, target_id)
                if previous_source != source_key or previous_target != target_id:
                    raise SourceCoverageUnavailable(
                        "CAD chart vertex correspondence is not a one-to-one source/target map.",
                        status="violated",
                    )
            oriented_sign = chart_sign * region_sign
            for edge in range(3):
                next_edge = (edge + 1) % 3
                first_id, second_id = targets[edge], targets[next_edge]
                first_key, second_key = labels[edge], labels[next_edge]
                if first_id == second_id or first_key == second_key:
                    raise SourceCoverageUnavailable(
                        "CAD chart maps a nondegenerate edge to one source vertex.",
                        status="violated",
                    )
                target_edge = (min(first_id, second_id), max(first_id, second_id))
                ordered_source_edge = tuple(sorted((first_key, second_key), key=repr))
                source_edge = (ordered_source_edge[0], ordered_source_edge[1])
                target_edges.setdefault((region, target_edge), []).append(row)
                source_edges.setdefault((region, source_edge), []).append(
                    (
                        row,
                        (1 if repr(first_key) < repr(second_key) else -1) * oriented_sign,
                    )
                )
                previous_edge = source_edge_for_target.setdefault(
                    (region, target_edge), source_edge
                )
                if previous_edge != source_edge:
                    raise SourceCoverageUnavailable(
                        "Adjacent CAD chart faces disagree on their common source edge.",
                        status="violated",
                    )

    topology: list[tuple[tuple[int, ...], int, int, int, int, int]] = []
    for region in range(len(domain.regions)):
        region_target_edges = {
            edge: rows for (owner, edge), rows in target_edges.items() if owner == region
        }
        region_source_edges = {
            edge: rows for (owner, edge), rows in source_edges.items() if owner == region
        }
        if not region_target_edges or not region_source_edges:
            raise SourceCoverageUnavailable(
                "A CAD material region has no charted target boundary.", status="violated"
            )
        for edge, rows in region_target_edges.items():
            if len(rows) != 2:
                raise SourceCoverageUnavailable(
                    "Target CAD chart boundary is not a closed two-manifold per material region.",
                    status="violated",
                )
            first_row, second_row = rows
            first_targets = row_targets[first_row]
            second_targets = row_targets[second_row]
            first_labels = dict(zip(first_targets, row_labels[first_row], strict=True))
            second_labels = dict(zip(second_targets, row_labels[second_row], strict=True))
            endpoints = edge
            if any(
                first_labels.get(value) != second_labels.get(value) for value in endpoints
            ):
                raise SourceCoverageUnavailable(
                    "Target faces reglue to different source chart points along a shared edge.",
                    status="violated",
                )
            first_source_edge = source_edge_for_target[(region, edge)]
            if first_source_edge not in region_source_edges:
                raise SourceCoverageUnavailable(
                    "Target CAD edge has no paired source chart edge.", status="violated"
                )
            bound_seam = ancestry is not None and (
                (first_row, edge) in ancestry.edge_uses
                or (second_row, edge) in ancestry.edge_uses
            )
            if patch_rows[first_row] != patch_rows[second_row] or bound_seam:
                first_patch, second_patch = patch_rows[first_row], patch_rows[second_row]
                first_uv = dict(zip(first_targets, row_uv[first_row], strict=True))
                second_uv = dict(zip(second_targets, row_uv[second_row], strict=True))
                if ancestry is not None:
                    first_use = ancestry.edge_uses.get((first_row, edge))
                    second_use = ancestry.edge_uses.get((second_row, edge))
                else:
                    first_use = (
                        _cad_plane_curve_use(
                            domain,
                            first_patch,
                            first_labels[endpoints[0]],
                            first_labels[endpoints[1]],
                        )
                        if first_patch in plane_patches
                        else _cad_chart_curve_use(
                            domain,
                            first_patch,
                            first_uv[endpoints[0]],
                            first_uv[endpoints[1]],
                        )
                    )
                    second_use = (
                        _cad_plane_curve_use(
                            domain,
                            second_patch,
                            second_labels[endpoints[0]],
                            second_labels[endpoints[1]],
                        )
                        if second_patch in plane_patches
                        else _cad_chart_curve_use(
                            domain,
                            second_patch,
                            second_uv[endpoints[0]],
                            second_uv[endpoints[1]],
                        )
                    )
                _cad_validate_coedge_pair(
                    first_use,
                    second_use,
                    dict(signs_by_patch[first_patch]).get(region, 0),
                    dict(signs_by_patch[second_patch]).get(region, 0),
                )
        for edge, owners in region_source_edges.items():
            if len(owners) != 2 or sum(sign for _, sign in owners) != 0:
                raise SourceCoverageUnavailable(
                    "Mapped CAD chart triangles do not form a closed oriented source-region 2-chain.",
                    status="violated",
                )
        reverse_edges: dict[tuple[_VertexKey, _VertexKey], tuple[int, int]] = {}
        for target_edge, source_edge in (
            (edge, source_edge_for_target[(region, edge)]) for edge in region_target_edges
        ):
            previous = reverse_edges.setdefault(source_edge, target_edge)
            if previous != target_edge:
                raise SourceCoverageUnavailable(
                    "Distinct target edges map to one authored CAD chart edge.",
                    status="violated",
                )
        if set(reverse_edges) != set(region_source_edges) or len(reverse_edges) != len(
            region_target_edges
        ):
            raise SourceCoverageUnavailable(
                "Target/source CAD edge correspondence is not bijective.",
                status="violated",
            )
        target_vertices = target_to_source.get(region, {})
        source_vertices = source_to_target.get(region, {})
        if len(target_vertices) != len(source_vertices) or len(target_vertices) != len(
            set(target_vertices)
        ):
            raise SourceCoverageUnavailable(
                "Target/source CAD vertex correspondence is not bijective.",
                status="violated",
            )
        face_rows = tuple(
            row for row, patch in enumerate(patch_rows) if region in pairs[patch]
        )
        target_euler = len(target_vertices) - len(region_target_edges) + len(face_rows)
        source_euler = len(source_vertices) - len(region_source_edges) + len(face_rows)
        if target_euler != source_euler:
            raise SourceCoverageUnavailable(
                "Target and authored CAD chart complexes have different Euler characteristics.",
                status="violated",
            )
        target_neighbors = {row: set() for row in face_rows}
        source_neighbors = {row: set() for row in face_rows}
        for rows in region_target_edges.values():
            target_neighbors[rows[0]].add(rows[1])
            target_neighbors[rows[1]].add(rows[0])
        for rows in region_source_edges.values():
            first_row, second_row = rows[0][0], rows[1][0]
            source_neighbors[first_row].add(second_row)
            source_neighbors[second_row].add(first_row)

        def components(graph: dict[int, set[int]]) -> tuple[tuple[int, ...], ...]:
            remaining = set(graph)
            result = []
            while remaining:
                stack = [remaining.pop()]
                component = set(stack)
                while stack:
                    for neighbor in graph[stack.pop()] & remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
                result.append(tuple(sorted(component)))
            return tuple(sorted(result))

        target_components = components(target_neighbors)
        source_components = components(source_neighbors)
        if target_neighbors != source_neighbors or target_components != source_components:
            raise SourceCoverageUnavailable(
                "Target and authored CAD chart components or edge incidence do not correspond.",
                status="violated",
            )
        if set(row_degree_signs.get(region, {})) != set(face_rows):
            raise SourceCoverageUnavailable(
                "CAD material-region degree omits a target chart face.", status="violated"
            )
        component_degrees = tuple(
            row_degree_signs[region][component[0]] for component in target_components
        )
        if any(value != 1 for value in component_degrees):
            raise SourceCoverageUnavailable(
                "CAD target/source boundary component degree is not one.",
                status="violated",
            )
        topology.append(
            (
                component_degrees,
                len(target_components),
                target_euler,
                len(target_vertices),
                len(region_target_edges),
                len(face_rows),
            )
        )
    return tuple(topology)


def _cad_chart_partition(
    domain: MeshingDomain,
    source: SourceBoundaryQuery,
    witness: SourceBoundaryChartWitness,
    mesh: CellMesh,
    cell_regions: np.ndarray,
    /,
) -> tuple[int, tuple[str, ...]]:
    from ._domain_boundary import exact_plane_chart, MeshingDomainBoundarySource
    from ._meshing_domain import SourcePreparation

    if not isinstance(source, MeshingDomainBoundarySource) or not _same_domain(
        source.domain, domain
    ):
        raise SourceCoverageUnavailable(
            "CAD approximate coverage requires the actual revision-bound MeshingDomain boundary query.",
            status="unresolved",
        )
    source.require_current()
    if domain.accuracy != "exact_represented" or domain.tolerance != 0.0:
        raise SourceCoverageUnavailable(
            "Approximate coverage of a toleranced CAD boundary requires authenticated incidence ribbons to a closed representative.",
            status="unresolved",
        )
    if set(source.patches) != set(range(len(domain.patches))):
        raise SourceCoverageUnavailable(
            "CAD coverage query omits source patches.", status="violated"
        )
    if witness.chart_dimension != 2 or witness.reference_dimension != 3:
        raise SourceCoverageUnavailable(
            "CAD volume chart witness has the wrong dimensions.", status="unresolved"
        )
    if (
        witness.source_id != domain.source_id
        or witness.source_revision != domain.source_revision
    ):
        raise SourceCoverageUnavailable(
            "CAD chart witness binds another source revision.", status="violated"
        )
    if witness.source_query_id != source.source_scope_id:
        raise SourceCoverageUnavailable(
            "CAD chart witness binds another source-boundary query scope.",
            status="violated",
        )
    pairs, signs_by_patch = cad_material_sides(domain)
    rooted = False
    try:
        declared_by_patch = dict(source.boundary_parameter_triangles())
    except (ArithmeticError, NotImplementedError, ValueError):
        if witness.cad_plc is None:
            raise
        # An uncertain source root is not an exact trim vertex. The retained
        # source charts must instead supply the independent rooted proof below.
        declared_by_patch = {}
        rooted = True
    targets_by_patch: dict[int, list[_Triangle2]] = {}
    plane_by_patch: dict[int, bool] = {}
    patch_rows: list[int] = []
    for row, (source_chart_id, source_entity_id, occurrence) in enumerate(
        zip(
            witness.source_chart_ids,
            witness.source_entity_ids,
            witness.occurrence_paths,
            strict=True,
        )
    ):
        try:
            patch = domain.resolve_entity_id(
                2, source_entity_id, occurrence_path=occurrence
            )
        except (TypeError, ValueError) as error:
            raise SourceCoverageUnavailable(
                f"CAD chart witness names an invalid source face: {error}",
                status="violated",
            ) from error
        plane = source_chart_id != source_entity_id
        if plane:
            plane_chart = exact_plane_chart(domain, patch)
            if plane_chart is None or source_chart_id != plane_chart.chart_id:
                raise SourceCoverageUnavailable(
                    "CAD source chart and authored face identity differ.",
                    status="violated",
                )
        if plane_by_patch.setdefault(patch, plane) != plane:
            raise SourceCoverageUnavailable(
                "CAD chart witness mixes charts of one source face.", status="violated"
            )
        patch_rows.append(patch)
        triangle = _triangle2(witness.source_parameter_chart(row))
        targets_by_patch.setdefault(patch, []).append(triangle)
    if set(targets_by_patch) != set(source.patches):
        raise SourceCoverageUnavailable(
            "CAD boundary chart witness omits or adds an authored source patch.",
            status="violated",
        )
    ancestry = None
    if rooted:
        from .brep._source_shell_admissibility import certify_single_source_shell

        plc = witness.cad_plc
        if (
            plc is None
            or plc.cad_source_charts is None
            or plc.cad_fidelity_tolerance is None
        ):
            raise SourceCoverageUnavailable(
                "Rooted CAD coverage requires retained source charts and fidelity premises.",
                status="unresolved",
            )
        ancestry = _cad_target_root_partition(
            domain, mesh, witness, patch_rows, plc.cad_fidelity_tolerance
        )
        if (
            certify_single_source_shell(plc.cad_source_charts, plc.cad_fidelity_tolerance)
            is None
        ):
            raise SourceCoverageUnavailable(
                "Rooted CAD source shell embedding or material orientation remains unresolved.",
                status="unresolved",
            )
    preparation = SourcePreparation(source.domain)
    for patch, plane in () if rooted else plane_by_patch.items():
        if plane:
            # A curved plane-chart trim is referenced by the target's boundary
            # chords, each separated from its source arc by a certified lens.
            declared_by_patch[patch] = source._trim_partition_reference_unchecked(
                patch, tuple(targets_by_patch[patch]), preparation, plane=True
            )[0]
    plane_patches = frozenset(patch for patch, plane in plane_by_patch.items() if plane)
    for patch in () if rooted else source.patches:
        declared = tuple(declared_by_patch[patch])
        target = tuple(targets_by_patch.get(patch, ()))
        if patch not in plane_patches:
            # Collapsed pole slivers are part of their owner cells' chart preimages.
            target = (
                *target,
                *(sliver for _, sliver in source.collapsed_pole_slivers(patch, target)),
            )
        if not declared or not target:
            raise SourceCoverageUnavailable(
                f"CAD patch {patch} lacks a complete chart partition.", status="violated"
            )
        source_area = sum((abs(_area2(triangle)) for triangle in declared), Fraction(0))
        target_area = sum((abs(_area2(triangle)) for triangle in target), Fraction(0))
        if target_area != source_area:
            raise SourceCoverageUnavailable(
                f"CAD source chart area differs on patch {patch}.", status="violated"
            )
        declared_boxes = tuple(_triangle_box(triangle) for triangle in declared)
        target_boxes = tuple(_triangle_box(triangle) for triangle in target)
        # Exact areas only for pairs the binary64 box and edge-line screens leave
        # possibly overlapping; every other pair has zero intersection area.
        declared_rows: list[list[int]] = [[] for _ in target]
        for first, value in _possibly_overlapping_pairs(target, declared, False):
            declared_rows[first].append(value)
        covered = Fraction(0)
        for first, box, rows in zip(target, target_boxes, declared_rows, strict=True):
            area = abs(_area2(first))
            if area == 0:
                raise SourceCoverageUnavailable(
                    f"CAD source chart is degenerate on patch {patch}.", status="violated"
                )
            inside = sum(
                (
                    _intersection_area(first, declared[row])
                    for row in rows
                    if _boxes_overlap(box, declared_boxes[row])
                ),
                Fraction(0),
            )
            if inside != area:
                raise SourceCoverageUnavailable(
                    f"CAD target chart leaves the authored trim on patch {patch}.",
                    status="violated",
                )
            covered += inside
        for index, other in _possibly_overlapping_pairs(target, target, True):
            if _boxes_overlap(target_boxes[index], target_boxes[other]) and (
                _intersection_area(target[index], target[other]) > 0
            ):
                raise SourceCoverageUnavailable(
                    f"CAD target charts overlap on patch {patch}.", status="violated"
                )
        if covered != source_area:
            raise SourceCoverageUnavailable(
                f"CAD chart partition does not cover patch {patch}.", status="violated"
            )

    owners, by_cell_face, face_outward_sign = _volume_face_owners(mesh, cell_regions)
    required_keys = {
        key
        for key, adjacent in owners.items()
        if len(adjacent) == 1 or adjacent[0][1] != adjacent[1][1]
    }
    witnessed_keys: list[tuple[int, ...]] = []
    cell_region = {
        cell_id: region for adjacent in owners.values() for cell_id, region, _ in adjacent
    }
    for row, (cell_id_raw, local_face_raw, patch) in enumerate(
        zip(
            witness.cell_global_ids, witness.local_entity_indices, patch_rows, strict=True
        )
    ):
        cell_id, local_face = int(cell_id_raw), int(local_face_raw)
        key = by_cell_face.get((cell_id, local_face))
        if key is None or key not in required_keys:
            raise SourceCoverageUnavailable(
                "CAD witness row is not a target boundary/interface face.",
                status="violated",
            )
        witnessed_keys.append(key)
        pair = pairs[patch]
        adjacent = owners[key]
        actual_regions = {region for _, region, _ in adjacent}
        expected_regions = {region for region in pair if region >= 0}
        if actual_regions != expected_regions or (-1 in pair) != (len(adjacent) == 1):
            raise SourceCoverageUnavailable(
                "CAD target face has wrong source material sides.", status="violated"
            )
        parent_region = cell_region[cell_id]
        source_orientation = dict(signs_by_patch[patch]).get(parent_region, 0)
        target_outward_sign = face_outward_sign[(cell_id, local_face)]
        if (
            source_orientation == 0
            or int(witness.orientations[row]) * target_outward_sign != source_orientation
        ):
            raise SourceCoverageUnavailable(
                "CAD chart orientation disagrees with the authored region boundary.",
                status="violated",
            )
    if set(witnessed_keys) != required_keys:
        raise SourceCoverageUnavailable(
            "CAD chart witness does not cover every target boundary/interface face.",
            status="violated",
        )
    topology = _cad_chart_topology_degree(
        domain,
        mesh,
        witness,
        patch_rows,
        pairs,
        signs_by_patch,
        by_cell_face,
        cell_region,
        face_outward_sign,
        plane_patches,
        ancestry,
    )
    degrees = tuple(
        canonical_fingerprint(
            {
                "kind": "cad-region-oriented-chart-degree",
                "domain": domain.domain_id,
                "source": (domain.source_id, domain.source_revision, domain.authority_id),
                "region": (
                    domain.region_source_indices[region_row],
                    domain.region_source_occurrences[region_row],
                    domain.regions[region_row].name,
                ),
                "oriented_source_patches": tuple(
                    sorted(
                        (
                            domain.entity_id(2, patch),
                            domain.source_occurrences[2][patch],
                            domain.patches[patch].orientation * orientation,
                        )
                        for patch, orientation in domain.regions[region_row].boundary
                    )
                ),
                "computed_degree": topology[region_row][0],
                "components": topology[region_row][1],
                "euler_characteristic": topology[region_row][2],
                "simplex_counts": topology[region_row][3:],
                "chart_witness": witness.witness_id,
            }
        )
        for region_row in range(len(domain.regions))
    )
    count = (
        len(witness.cad_plc.cad_source_charts.cell_global_ids)
        if rooted
        and witness.cad_plc is not None
        and witness.cad_plc.cad_source_charts is not None
        else sum(len(triangles) for triangles in declared_by_patch.values())
    )
    return count, degrees


def _same_domain(first: MeshingDomain, second: MeshingDomain, /) -> bool:
    from ._mesh_certificates import _same_tree_content

    return first.domain_id == second.domain_id and _same_tree_content(first, second)


def _interval_product(
    first: tuple[Fraction, Fraction], second: tuple[Fraction, Fraction], /
) -> tuple[Fraction, Fraction]:
    values = (
        first[0] * second[0],
        first[0] * second[1],
        first[1] * second[0],
        first[1] * second[1],
    )
    return min(values), max(values)


def _interval_dot(
    gradient_lower: np.ndarray,
    gradient_upper: np.ndarray,
    direction: tuple[Fraction, Fraction, Fraction],
    /,
) -> tuple[Fraction, Fraction]:
    lower = Fraction(0)
    upper = Fraction(0)
    for axis, component in enumerate(direction):
        product = _interval_product(
            (_fraction(gradient_lower[axis]), _fraction(gradient_upper[axis])),
            (component, component),
        )
        lower += product[0]
        upper += product[1]
    return lower, upper


def _strict_sign(interval: tuple[Fraction, Fraction], /) -> int:
    return 1 if interval[0] > 0 else -1 if interval[1] < 0 else 0


def _interval_subtract(
    first: tuple[Fraction, Fraction], second: tuple[Fraction, Fraction], /
) -> tuple[Fraction, Fraction]:
    return first[0] - second[1], first[1] - second[0]


def _root_component_interval(
    scaffold: AdaptiveImplicitScaffold, root_row: int, axis: int, /
) -> tuple[Fraction, Fraction]:
    bracket = np.asarray(scaffold.root_brackets[root_row], dtype=np.float64)
    first, second = _fraction(bracket[0, axis]), _fraction(bracket[1, axis])
    return min(first, second), max(first, second)


def _root_triangle_projected_area(
    scaffold: AdaptiveImplicitScaffold, root_rows: tuple[int, int, int], axis: int, /
) -> tuple[Fraction, Fraction]:
    if axis == 0:
        first_axis, second_axis = 1, 2
    elif axis == 1:
        first_axis, second_axis = 0, 2
    elif axis == 2:
        first_axis, second_axis = 0, 1
    else:
        raise ValueError("A projected root triangle axis must be three-dimensional.")
    points = (
        (
            _root_component_interval(scaffold, root_rows[0], first_axis),
            _root_component_interval(scaffold, root_rows[0], second_axis),
        ),
        (
            _root_component_interval(scaffold, root_rows[1], first_axis),
            _root_component_interval(scaffold, root_rows[1], second_axis),
        ),
        (
            _root_component_interval(scaffold, root_rows[2], first_axis),
            _root_component_interval(scaffold, root_rows[2], second_axis),
        ),
    )
    first = _interval_subtract(points[1][0], points[0][0])
    second = _interval_subtract(points[1][1], points[0][1])
    third = _interval_subtract(points[2][0], points[0][0])
    fourth = _interval_subtract(points[2][1], points[0][1])
    return _interval_subtract(
        _interval_product(first, fourth), _interval_product(third, second)
    )


def _root_edge_side_interval(
    scaffold: AdaptiveImplicitScaffold, first: int, second: int, point: int, axis: int, /
) -> tuple[Fraction, Fraction]:
    if axis == 0:
        first_axis, second_axis = 1, 2
    elif axis == 1:
        first_axis, second_axis = 0, 2
    elif axis == 2:
        first_axis, second_axis = 0, 1
    else:
        raise ValueError("A projected root edge axis must be three-dimensional.")
    first_point = (
        _root_component_interval(scaffold, first, first_axis),
        _root_component_interval(scaffold, first, second_axis),
    )
    second_point = (
        _root_component_interval(scaffold, second, first_axis),
        _root_component_interval(scaffold, second, second_axis),
    )
    test_point = (
        _root_component_interval(scaffold, point, first_axis),
        _root_component_interval(scaffold, point, second_axis),
    )
    first_x = _interval_subtract(second_point[0], first_point[0])
    first_y = _interval_subtract(second_point[1], first_point[1])
    second_x = _interval_subtract(test_point[0], first_point[0])
    second_y = _interval_subtract(test_point[1], first_point[1])
    return _interval_subtract(
        _interval_product(first_x, second_y), _interval_product(first_y, second_x)
    )


def _validate_oriented_triangle_complex(faces: np.ndarray, /) -> None:
    face_rows = np.asarray(faces, dtype=np.int64)
    if face_rows.ndim != 2 or face_rows.shape[1] != 3:
        raise SourceCoverageUnavailable(
            "Implicit source boundary is not triangular.", status="unresolved"
        )
    edge_incidence: dict[tuple[int, int], tuple[int, int]] = {}
    triangle_keys: set[tuple[int, int, int]] = set()
    for row in range(face_rows.shape[0]):
        face = face_rows[row]
        triangle = (int(face[0]), int(face[1]), int(face[2]))
        ordered = sorted(triangle)
        key = (ordered[0], ordered[1], ordered[2])
        if len(set(triangle)) != 3 or key in triangle_keys:
            raise SourceCoverageUnavailable(
                "Implicit source boundary repeats or degenerates a triangle.",
                status="violated",
            )
        triangle_keys.add(key)
        for first, second in zip(
            triangle, (triangle[1], triangle[2], triangle[0]), strict=True
        ):
            edge = (min(first, second), max(first, second))
            count, balance = edge_incidence.get(edge, (0, 0))
            edge_incidence[edge] = (count + 1, balance + (1 if first < second else -1))
    if not edge_incidence or any(
        count != 2 or balance != 0 for count, balance in edge_incidence.values()
    ):
        raise SourceCoverageUnavailable(
            "Implicit source root triangles are not a closed oriented 2-chain.",
            status="violated",
        )


def _implicit_root_derivation(
    surface: AdaptiveImplicitSurface,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_regions: np.ndarray,
    witness: SourceBoundaryChartWitness,
    /,
) -> tuple[str, int]:
    from ._certificate import FieldRegularity
    from ._contracts import GeometryKind
    from .implicit._adaptive_discovery import discover_adaptive_implicit_surface

    if not isinstance(surface, AdaptiveImplicitSurface):
        raise SourceCoverageUnavailable(
            "Implicit coverage requires AdaptiveImplicitSurface.", status="unresolved"
        )

    if (
        surface.geometry.kind is not GeometryKind.REGION
        or surface.policy.allow_nonsmooth_field
        or surface.geometry.field_certificate.regularity is not FieldRegularity.SMOOTH
    ):
        raise SourceCoverageUnavailable(
            "Implicit coverage requires a smooth region-valued source.",
            status="unresolved",
        )
    recipe = surface.archive_recipe
    fresh = discover_adaptive_implicit_surface(
        recipe.geometry,
        domain=recipe.domain,
        policy=recipe.policy,
        source_id=recipe.source_id,
    )
    fresh_cover = fresh.cover
    surface_cover = surface.cover
    fresh_mesh = fresh.mesh
    surface_mesh = surface.mesh
    if (
        fresh.domain_id != surface.domain_id
        or fresh.evidence.evidence_id != surface.evidence.evidence_id
        or fresh_cover is None
        or surface_cover is None
        or fresh_cover.cover_id != surface_cover.cover_id
        or fresh.volume.query_id != surface.volume.query_id
        or array_tree_fingerprint(fresh.scaffold)
        != array_tree_fingerprint(surface.scaffold)
        or fresh_mesh is None
        or surface_mesh is None
        or array_tree_fingerprint((fresh_mesh.vertices, fresh_mesh.faces))
        != array_tree_fingerprint((surface_mesh.vertices, surface_mesh.faces))
    ):
        raise SourceCoverageUnavailable(
            "Implicit discovery does not cold-reproduce its retained roots and intervals.",
            status="violated",
        )
    cover = surface_cover
    scaffold = surface.scaffold
    source_mesh = fresh_mesh
    bounds = fresh.volume.bounds
    if (
        not bounds.value_rigorous
        or not bounds.gradient_rigorous
        or fresh.evidence.unresolved_issues.size
    ):
        raise SourceCoverageUnavailable(
            "Implicit source has unresolved value, gradient, root, or topology premises.",
            status="unresolved",
        )
    _validate_oriented_triangle_complex(np.asarray(source_mesh.faces, dtype=np.int64))
    if (
        scaffold.cell_indices.shape[0] != scaffold.cell_boxes.shape[0]
        or scaffold.triangle_cell_rows.shape[0] != source_mesh.faces.shape[0]
        or scaffold.triangle_root_edge_rows.shape[0] != source_mesh.faces.shape[0]
        or np.any(scaffold.cell_issues != 0)
        or np.any(scaffold.root_issues != 0)
        or np.any(scaffold.corner_signs == 0)
        or np.any(scaffold.root_errors < 0.0)
    ):
        raise SourceCoverageUnavailable(
            "Implicit source root ancestry contains unresolved strata.",
            status="unresolved",
        )

    # Reprove a nonoverlapping cover of the full source box and its value enclosures.
    boxes = np.asarray(cover.boxes, dtype=np.float64)
    if boxes.shape[0] == 0:
        raise SourceCoverageUnavailable(
            "Implicit source has no interval cover leaves.", status="unresolved"
        )
    domain_box = np.asarray(cover.domain, dtype=np.float64)
    exact_domain_volume = math.prod(
        _fraction(domain_box[1, axis]) - _fraction(domain_box[0, axis])
        for axis in range(3)
    )
    cover_volume = Fraction(0)
    for lower, upper in boxes:
        if np.any(lower < domain_box[0]) or np.any(upper > domain_box[1]):
            raise SourceCoverageUnavailable(
                "Implicit cover leaves escape the declared source box.", status="violated"
            )
        cover_volume += math.prod(
            _fraction(upper[axis]) - _fraction(lower[axis]) for axis in range(3)
        )
    order = np.argsort(boxes[:, 0, 0], kind="stable")
    active: list[int] = []
    for raw_row in order:
        row = int(raw_row)
        lower, upper = boxes[row]
        active = [other for other in active if boxes[other, 1, 0] > lower[0]]
        for other in active:
            other_lower, other_upper = boxes[other]
            if min(upper[1], other_upper[1]) > max(lower[1], other_lower[1]) and min(
                upper[2], other_upper[2]
            ) > max(lower[2], other_lower[2]):
                raise SourceCoverageUnavailable(
                    "Implicit interval cover has overlapping leaf interiors.",
                    status="violated",
                )
        active.append(row)
    if cover_volume != exact_domain_volume:
        raise SourceCoverageUnavailable(
            "Implicit interval leaves do not cover the full source box.",
            status="violated",
        )
    for start in range(0, boxes.shape[0], 256):
        stop = min(start + 256, boxes.shape[0])
        evaluated = bounds.boxes(boxes[start:stop, 0], boxes[start:stop, 1])
        if np.any(cover.value_lower[start:stop] > evaluated.value_lower) or np.any(
            cover.value_upper[start:stop] < evaluated.value_upper
        ):
            raise SourceCoverageUnavailable(
                "Implicit cover does not retain valid source value enclosures.",
                status="violated",
            )

    def box_key(box: np.ndarray, /) -> tuple[_Point3, _Point3]:
        return (
            _point3(tuple(_fraction(value) for value in box[0])),
            _point3(tuple(_fraction(value) for value in box[1])),
        )

    possible_zero_boxes = {
        box_key(boxes[row])
        for row in range(boxes.shape[0])
        if cover.value_lower[row] <= 0.0 <= cover.value_upper[row]
    }
    scaffold_boxes = {box_key(box) for box in np.asarray(scaffold.cell_boxes)}
    if possible_zero_boxes != scaffold_boxes:
        raise SourceCoverageUnavailable(
            "The complete implicit interval cover and root-cell scaffold disagree on possible-zero strata.",
            status="violated",
        )

    corner_rows_array = np.asarray(scaffold.corner_ids, dtype=np.int64)
    if corner_rows_array.ndim != 2 or corner_rows_array.shape[1] != 3:
        raise SourceCoverageUnavailable(
            "Implicit grid corner IDs are not three-dimensional.", status="violated"
        )
    corner_ids: tuple[_GridPoint, ...] = tuple(
        (int(row[0]), int(row[1]), int(row[2])) for row in corner_rows_array
    )
    corner_rows: dict[_GridPoint, int] = {
        identifier: row for row, identifier in enumerate(corner_ids)
    }
    root_edge_rows = np.asarray(scaffold.root_edge_ids, dtype=np.int64)
    if root_edge_rows.ndim != 3 or root_edge_rows.shape[1:] != (2, 3):
        raise SourceCoverageUnavailable(
            "Implicit root edge IDs omit endpoint coordinates.", status="violated"
        )
    root_keys_list: list[tuple[_GridPoint, _GridPoint]] = []
    for row in range(root_edge_rows.shape[0]):
        edge = root_edge_rows[row]
        first = (int(edge[0, 0]), int(edge[0, 1]), int(edge[0, 2]))
        second = (int(edge[1, 0]), int(edge[1, 1]), int(edge[1, 2]))
        if first == second:
            raise SourceCoverageUnavailable(
                "Implicit root edge has identical grid endpoints.", status="violated"
            )
        root_keys_list.append((first, second) if first < second else (second, first))
    root_keys = tuple(root_keys_list)
    root_by_key: dict[tuple[_GridPoint, _GridPoint], int] = {
        key: row for row, key in enumerate(root_keys)
    }
    if len(root_by_key) != len(root_keys):
        raise SourceCoverageUnavailable(
            "Implicit scaffold repeats a source root edge.", status="violated"
        )
    root_by_vertex: dict[_Point3, int] = {}
    root_point_rows = np.asarray(scaffold.root_points, dtype=np.float64)
    for root_row in range(root_point_rows.shape[0]):
        key = _point3(tuple(_fraction(value) for value in root_point_rows[root_row]))
        if key in root_by_vertex:
            raise SourceCoverageUnavailable(
                "Implicit scaffold merges distinct source root identities.",
                status="violated",
            )
        root_by_vertex[key] = root_row

    def certify_zero_free_segment(first: np.ndarray, second: np.ndarray, /) -> bool:
        stack = [(np.array(first, copy=True), np.array(second, copy=True))]
        while stack:
            low_point, high_point = stack.pop()
            lower = np.minimum(low_point, high_point)[None, :]
            upper = np.maximum(low_point, high_point)[None, :]
            enclosure = bounds.boxes(lower, upper)
            if enclosure.value_lower[0] > 0.0 or enclosure.value_upper[0] < 0.0:
                continue
            midpoint = low_point + 0.5 * (high_point - low_point)
            if np.array_equal(midpoint, low_point) or np.array_equal(
                midpoint, high_point
            ):
                return False
            stack.append((midpoint, high_point))
            stack.append((low_point, midpoint))
        return True

    # Each retained root is the unique simple root on its complete source edge.
    for root_row, edge in enumerate(root_keys):
        if edge[0] not in corner_rows or edge[1] not in corner_rows:
            raise SourceCoverageUnavailable(
                "Implicit root ancestry references absent grid corners.",
                status="violated",
            )
        first = np.asarray(
            scaffold.corner_coordinates[corner_rows[edge[0]]], dtype=np.float64
        )
        second = np.asarray(
            scaffold.corner_coordinates[corner_rows[edge[1]]], dtype=np.float64
        )
        bracket = np.asarray(scaffold.root_brackets[root_row], dtype=np.float64)
        point = np.asarray(scaffold.root_points[root_row], dtype=np.float64)
        bracket_lower = np.minimum(bracket[0], bracket[1])
        bracket_upper = np.maximum(bracket[0], bracket[1])
        if np.any(point < bracket_lower) or np.any(point > bracket_upper):
            raise SourceCoverageUnavailable(
                "Implicit root point escapes its retained bracket.", status="violated"
            )
        first_value = bounds.point_values(first[None, :])
        second_value = bounds.point_values(second[None, :])
        sign_first = (
            -1 if first_value[1][0] < 0.0 else 1 if first_value[0][0] > 0.0 else 0
        )
        sign_second = (
            -1 if second_value[1][0] < 0.0 else 1 if second_value[0][0] > 0.0 else 0
        )
        bracket_values = bounds.point_values(bracket)
        sign_low = (
            -1 if bracket_values[1][0] < 0.0 else 1 if bracket_values[0][0] > 0.0 else 0
        )
        sign_high = (
            -1 if bracket_values[1][1] < 0.0 else 1 if bracket_values[0][1] > 0.0 else 0
        )
        if (
            not sign_first
            or not sign_second
            or sign_first == sign_second
            or not sign_low
            or not sign_high
            or sign_low == sign_high
        ):
            raise SourceCoverageUnavailable(
                "Implicit root endpoints lack opposite strict interval signs.",
                status="unresolved",
            )
        lower_box = np.minimum(bracket[0], bracket[1])[None, :]
        upper_box = np.maximum(bracket[0], bracket[1])[None, :]
        derivative = bounds.boxes(lower_box, upper_box)
        direction = (
            _fraction(second[0]) - _fraction(first[0]),
            _fraction(second[1]) - _fraction(first[1]),
            _fraction(second[2]) - _fraction(first[2]),
        )
        derivative_sign = _strict_sign(
            _interval_dot(
                derivative.gradient_lower[0], derivative.gradient_upper[0], direction
            )
        )
        if derivative_sign == 0:
            raise SourceCoverageUnavailable(
                "Implicit root bracket is not proven simple and unique.",
                status="unresolved",
            )
        if not certify_zero_free_segment(
            first, bracket[0]
        ) or not certify_zero_free_segment(bracket[1], second):
            raise SourceCoverageUnavailable(
                "Implicit root edge has unexcluded additional zero roots.",
                status="unresolved",
            )

    # Every zero-bearing cell/tetrahedron is a regular graph and every face
    # restriction has only the root arcs retained by the tetrahedral ancestry.
    cell_gradient_axes: dict[int, int] = {}
    cell_boxes = np.asarray(scaffold.cell_boxes, dtype=np.float64)
    if cell_boxes.ndim != 3 or cell_boxes.shape[1:] != (2, 3):
        raise SourceCoverageUnavailable(
            "Implicit cell boxes do not retain three-dimensional interval bounds.",
            status="unresolved",
        )
    for cell_row in range(cell_boxes.shape[0]):
        lower = cell_boxes[cell_row, 0]
        upper = cell_boxes[cell_row, 1]
        enclosure = bounds.boxes(lower[None, :], upper[None, :])
        axes = tuple(
            axis
            for axis in range(3)
            if enclosure.gradient_lower[0, axis] > 0.0
            or enclosure.gradient_upper[0, axis] < 0.0
        )
        if not axes:
            raise SourceCoverageUnavailable(
                "Implicit surface cell has no certified monotone chart direction.",
                status="unresolved",
            )
        cell_gradient_axes[cell_row] = axes[0]

    assigned_triangle_rows: set[int] = set()
    tetrahedra = np.asarray(scaffold.tetrahedra, dtype=np.int64)
    tetra_cells = np.asarray(scaffold.tetrahedron_cell_rows, dtype=np.int64)
    triangle_cells = np.asarray(scaffold.triangle_cell_rows, dtype=np.int64)
    triangle_roots = np.asarray(scaffold.triangle_root_edge_rows, dtype=np.int64)
    signs = np.asarray(scaffold.corner_signs, dtype=np.int8)
    if (
        tetrahedra.ndim != 2
        or tetrahedra.shape[1] != 4
        or tetra_cells.shape != (tetrahedra.shape[0],)
        or triangle_roots.ndim != 2
        or triangle_roots.shape[1] != 3
        or triangle_cells.shape != (triangle_roots.shape[0],)
        or signs.ndim != 1
    ):
        raise SourceCoverageUnavailable(
            "Implicit primal/dual root ancestry has invalid fixed-dimensional arrays.",
            status="violated",
        )

    def triangle_root_ids(row: int, /) -> tuple[int, int, int]:
        return (
            int(triangle_roots[row, 0]),
            int(triangle_roots[row, 1]),
            int(triangle_roots[row, 2]),
        )

    if np.any(tetra_cells < 0) or np.any(tetra_cells >= cell_boxes.shape[0]):
        raise SourceCoverageUnavailable(
            "Implicit tetrahedron ancestry names an absent source cell.",
            status="violated",
        )
    for tetra_row in range(tetrahedra.shape[0]):
        cell_row = int(tetra_cells[tetra_row])
        corner_rows_tet: tuple[int, int, int, int] = (
            int(tetrahedra[tetra_row, 0]),
            int(tetrahedra[tetra_row, 1]),
            int(tetrahedra[tetra_row, 2]),
            int(tetrahedra[tetra_row, 3]),
        )
        grid_ids = tuple(corner_ids[row] for row in corner_rows_tet)
        tet_signs = tuple(int(signs[row]) for row in corner_rows_tet)
        crossing_roots: set[int] = set()
        for first_index in range(4):
            for second_index in range(first_index + 1, 4):
                if tet_signs[first_index] != tet_signs[second_index]:
                    key = tuple(sorted((grid_ids[first_index], grid_ids[second_index])))
                    root_row = root_by_key.get(key)
                    if root_row is None:
                        raise SourceCoverageUnavailable(
                            "Implicit tetrahedron crossing lacks a retained edge root.",
                            status="violated",
                        )
                    crossing_roots.add(root_row)
        if not crossing_roots:
            continue
        if len(crossing_roots) not in (3, 4):
            raise SourceCoverageUnavailable(
                "Implicit tetrahedron has an unresolved root polygon.",
                status="unresolved",
            )
        candidates = [
            row
            for row in range(triangle_roots.shape[0])
            if int(triangle_cells[row]) == cell_row
            and set(triangle_root_ids(row)).issubset(crossing_roots)
        ]
        expected_count = 1 if len(crossing_roots) == 3 else 2
        if len(candidates) != expected_count:
            raise SourceCoverageUnavailable(
                "Implicit root polygon does not have its complete source triangle ancestry.",
                status="violated",
            )
        for row in candidates:
            if row in assigned_triangle_rows:
                raise SourceCoverageUnavailable(
                    "Implicit source triangle is multiply assigned to tetrahedra.",
                    status="violated",
                )
            assigned_triangle_rows.add(row)
        if {
            value for row in candidates for value in triangle_root_ids(row)
        } != crossing_roots:
            raise SourceCoverageUnavailable(
                "Implicit source triangles omit a tetrahedral edge root.",
                status="violated",
            )
        if (
            len(candidates) == 2
            and len(
                set(triangle_root_ids(candidates[0]))
                & set(triangle_root_ids(candidates[1]))
            )
            != 2
        ):
            raise SourceCoverageUnavailable(
                "Implicit quadrilateral source chart has no shared diagonal.",
                status="violated",
            )
        graph_axis = cell_gradient_axes[cell_row]
        triangle_rows: tuple[tuple[int, int, int], ...] = tuple(
            triangle_root_ids(row) for row in candidates
        )
        projected_signs = tuple(
            _strict_sign(_root_triangle_projected_area(scaffold, triangle, graph_axis))
            for triangle in triangle_rows
        )
        if any(sign == 0 for sign in projected_signs) or len(set(projected_signs)) != 1:
            raise SourceCoverageUnavailable(
                "Implicit root triangles are not a regular graph in the certified cell chart.",
                status="unresolved",
            )
        if len(triangle_rows) == 2:
            shared = tuple(set(triangle_rows[0]) & set(triangle_rows[1]))
            first_only = tuple(set(triangle_rows[0]) - set(shared))
            second_only = tuple(set(triangle_rows[1]) - set(shared))
            if len(shared) != 2 or len(first_only) != 1 or len(second_only) != 1:
                raise SourceCoverageUnavailable(
                    "Implicit root quadrilateral is not a two-triangle disk.",
                    status="violated",
                )
            first_side = _strict_sign(
                _root_edge_side_interval(
                    scaffold, shared[0], shared[1], first_only[0], graph_axis
                )
            )
            second_side = _strict_sign(
                _root_edge_side_interval(
                    scaffold, shared[0], shared[1], second_only[0], graph_axis
                )
            )
            if first_side == 0 or second_side != -first_side:
                raise SourceCoverageUnavailable(
                    "Implicit root quadrilateral triangles overlap or fold.",
                    status="unresolved",
                )

        # The source field is monotone in a fixed cell direction. Restrict every
        # crossed tetrahedron face to a verified monotone tangential direction.
        for face in combinations(range(4), 3):
            face_signs = tuple(tet_signs[index] for index in face)
            crossings = sum(
                face_signs[index] != face_signs[(index + 1) % 3] for index in range(3)
            )
            face_points = np.asarray(
                [scaffold.corner_coordinates[corner_rows_tet[index]] for index in face],
                dtype=np.float64,
            )
            face_lower = np.min(face_points, axis=0)[None, :]
            face_upper = np.max(face_points, axis=0)[None, :]
            face_enclosure = bounds.boxes(face_lower, face_upper)
            if crossings == 0:
                if face_enclosure.value_lower[0] <= 0.0 <= face_enclosure.value_upper[0]:
                    raise SourceCoverageUnavailable(
                        "Implicit tetrahedral face may contain an untracked zero loop.",
                        status="unresolved",
                    )
            elif crossings == 2:
                tangents = (
                    (
                        _fraction(face_points[1, 0]) - _fraction(face_points[0, 0]),
                        _fraction(face_points[1, 1]) - _fraction(face_points[0, 1]),
                        _fraction(face_points[1, 2]) - _fraction(face_points[0, 2]),
                    ),
                    (
                        _fraction(face_points[2, 0]) - _fraction(face_points[1, 0]),
                        _fraction(face_points[2, 1]) - _fraction(face_points[1, 1]),
                        _fraction(face_points[2, 2]) - _fraction(face_points[1, 2]),
                    ),
                )
                if not any(
                    _strict_sign(
                        _interval_dot(
                            face_enclosure.gradient_lower[0],
                            face_enclosure.gradient_upper[0],
                            tangent,
                        )
                    )
                    for tangent in tangents
                ):
                    raise SourceCoverageUnavailable(
                        "Implicit face root arc is not stratifiably regular.",
                        status="unresolved",
                    )
            else:
                raise SourceCoverageUnavailable(
                    "Implicit tetrahedral face has ambiguous root incidence.",
                    status="unresolved",
                )
    if assigned_triangle_rows != set(range(source_mesh.faces.shape[0])):
        raise SourceCoverageUnavailable(
            "Implicit source triangles are not all bound to unique root tetrahedra.",
            status="violated",
        )

    # The root-point carrier must be an embedded, oriented subdivision of the
    # independently reconstructed source scaffold. Exact coordinates and root
    # ancestry, not source identity labels, establish the relation.
    source_vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
    source_face_rows = np.asarray(source_mesh.faces, dtype=np.int64)
    for triangle_row in range(source_face_rows.shape[0]):
        face = source_face_rows[triangle_row]
        rows = (
            root_by_vertex.get(
                _point3(
                    tuple(
                        _fraction(source_vertices[int(face[0]), axis])
                        for axis in range(3)
                    )
                )
            ),
            root_by_vertex.get(
                _point3(
                    tuple(
                        _fraction(source_vertices[int(face[1]), axis])
                        for axis in range(3)
                    )
                )
            ),
            root_by_vertex.get(
                _point3(
                    tuple(
                        _fraction(source_vertices[int(face[2]), axis])
                        for axis in range(3)
                    )
                )
            ),
        )
        first_root, second_root, third_root = rows
        if first_root is None or second_root is None or third_root is None:
            raise SourceCoverageUnavailable(
                "Implicit triangle ancestry does not reproduce its source root-edge vertices.",
                status="violated",
            )
        actual_rows = (first_root, second_root, third_root)
        expected_rows = (
            int(triangle_roots[triangle_row, 0]),
            int(triangle_roots[triangle_row, 1]),
            int(triangle_roots[triangle_row, 2]),
        )
        if actual_rows != expected_rows:
            raise SourceCoverageUnavailable(
                "Implicit triangle ancestry does not reproduce its source root-edge vertices.",
                status="violated",
            )
    target_faces, issues = _target_facets(mesh, geometry, cell_regions)
    if issues:
        raise SourceCoverageUnavailable(issues[0], status="unresolved")
    source_by_key: dict[_Triangle3, int] = {}
    source_triangles = source_vertices[source_face_rows]
    for row in range(source_triangles.shape[0]):
        triangle = _triangle3(
            tuple(
                tuple(_fraction(value) for value in source_triangles[row, vertex])
                for vertex in range(3)
            )
        )
        ordered = sorted(triangle)
        key = (ordered[0], ordered[1], ordered[2])
        if key in source_by_key:
            raise SourceCoverageUnavailable(
                "Implicit source scaffold repeats a target boundary triangle.",
                status="violated",
            )
        source_by_key[key] = row
    target_by_key: dict[_Triangle3, OrientedFacet] = {}
    for facet in target_faces:
        triangle = _triangle3(facet.vertices)
        ordered = sorted(triangle)
        key = (ordered[0], ordered[1], ordered[2])
        if key in target_by_key:
            raise SourceCoverageUnavailable(
                "Implicit target repeats a boundary triangle.", status="violated"
            )
        target_by_key[key] = facet
    if set(target_by_key) != set(source_by_key):
        raise SourceCoverageUnavailable(
            "Implicit target boundary is not an exact conforming subdivision of the source-root scaffold.",
            status="violated",
        )

    face_owners, face_by_cell, face_outward_sign = _volume_face_owners(mesh, cell_regions)
    required_face_keys = {
        key
        for key, adjacent in face_owners.items()
        if len(adjacent) == 1 or adjacent[0][1] != adjacent[1][1]
    }
    face_chart_keys = _target_face_chart_keys(mesh, geometry)
    witnessed_face_keys: set[tuple[int, ...]] = set()
    witnessed_triangles: set[int] = set()
    if (
        witness.implicit_cell_keys.shape != (witness.cell_global_ids.size, 4)
        or witness.implicit_root_edge_keys.shape
        != (witness.cell_global_ids.size, 3, 2, 3)
        or witness.cell_global_ids.size != len(required_face_keys)
    ):
        raise SourceCoverageUnavailable(
            "Implicit witness omits target faces or retained root ancestry.",
            status="violated",
        )
    for witness_row, (cell_id_raw, local_face_raw) in enumerate(
        zip(witness.cell_global_ids, witness.local_entity_indices, strict=True)
    ):
        cell_id, local_face = int(cell_id_raw), int(local_face_raw)
        face_key = face_by_cell.get((cell_id, local_face))
        if (
            face_key is None
            or face_key not in required_face_keys
            or face_key in witnessed_face_keys
        ):
            raise SourceCoverageUnavailable(
                "Implicit witness repeats or omits a target boundary facet.",
                status="violated",
            )
        witnessed_face_keys.add(face_key)
        target_triangle_key = face_chart_keys.get((cell_id, local_face))
        if target_triangle_key is None:
            raise SourceCoverageUnavailable(
                "Implicit target face has no retained exact chart key.", status="violated"
            )
        source_row = source_by_key.get(target_triangle_key)
        if source_row is None or source_row in witnessed_triangles:
            raise SourceCoverageUnavailable(
                "Implicit target facet has no unique source root triangle.",
                status="violated",
            )
        witnessed_triangles.add(source_row)
        source_cell = int(triangle_cells[source_row])
        expected_cell_key = tuple(
            int(value) for value in scaffold.cell_indices[source_cell]
        )
        if (
            tuple(int(value) for value in witness.implicit_cell_keys[witness_row])
            != expected_cell_key
        ):
            raise SourceCoverageUnavailable(
                "Implicit witness changes its source root-cell ancestry.",
                status="violated",
            )
        expected_edge_keys = (
            root_keys[int(triangle_roots[source_row, 0])],
            root_keys[int(triangle_roots[source_row, 1])],
            root_keys[int(triangle_roots[source_row, 2])],
        )
        actual_edge_rows = np.asarray(
            witness.implicit_root_edge_keys[witness_row], dtype=np.int64
        )
        if actual_edge_rows.shape != (3, 2, 3):
            raise SourceCoverageUnavailable(
                "Implicit witness root edges omit endpoint ancestry.", status="violated"
            )
        actual_edge_keys_list: list[tuple[_GridPoint, _GridPoint]] = []
        for edge_row in range(3):
            edge = actual_edge_rows[edge_row]
            first = (int(edge[0, 0]), int(edge[0, 1]), int(edge[0, 2]))
            second = (int(edge[1, 0]), int(edge[1, 1]), int(edge[1, 2]))
            actual_edge_keys_list.append(
                (first, second) if first < second else (second, first)
            )
        if set(actual_edge_keys_list) != set(expected_edge_keys):
            raise SourceCoverageUnavailable(
                "Implicit witness changes its source root-edge ancestry.",
                status="violated",
            )
        if (
            int(witness.orientations[witness_row])
            * face_outward_sign[(cell_id, local_face)]
            != 1
        ):
            raise SourceCoverageUnavailable(
                "Implicit boundary chart reverses the source outward orientation.",
                status="violated",
            )
        source_order = _triangle3(
            tuple(
                tuple(_fraction(value) for value in source_triangles[source_row, vertex])
                for vertex in range(3)
            )
        )
        target_facet = target_by_key[target_triangle_key]
        target_order = _triangle3(target_facet.vertices)
        source_normal = _cross3(source_order)
        target_normal = _cross3(target_order)
        if _dot3(source_normal, target_normal) <= 0:
            raise SourceCoverageUnavailable(
                "Implicit target boundary winding reverses a source root triangle.",
                status="violated",
            )
        gradient = bounds.boxes(
            np.min(np.asarray(target_order, dtype=np.float64), axis=0)[None, :],
            np.max(np.asarray(target_order, dtype=np.float64), axis=0)[None, :],
        )
        normal_interval = _interval_dot(
            gradient.gradient_lower[0], gradient.gradient_upper[0], target_normal
        )
        if _strict_sign(normal_interval) <= 0:
            raise SourceCoverageUnavailable(
                "Implicit source boundary normal is not certified outward on a root triangle.",
                status="unresolved",
            )
    if witnessed_face_keys != required_face_keys or witnessed_triangles != set(
        range(source_mesh.faces.shape[0])
    ):
        raise SourceCoverageUnavailable(
            "Implicit root chart witness does not cover the complete oriented boundary.",
            status="violated",
        )

    degree = canonical_fingerprint(
        {
            "kind": "implicit-source-stratified-boundary-degree",
            "domain": surface.domain_id,
            "source": (surface.source_id, fresh.volume.query_id),
            "region_id": surface.region_ids[0],
            "root_cells": array_tree_fingerprint(
                (scaffold.cell_indices, scaffold.tetrahedra)
            ),
            "root_intervals": array_tree_fingerprint(
                (
                    scaffold.root_edge_ids,
                    scaffold.root_brackets,
                    scaffold.root_endpoint_value_lower,
                    scaffold.root_endpoint_value_upper,
                )
            ),
            "triangle_ancestry": array_tree_fingerprint(
                (scaffold.triangle_cell_rows, scaffold.triangle_root_edge_rows)
            ),
            "witness": witness.witness_id,
            "orientation": "gradient-outward-negative-interior",
        }
    )
    return degree, source_mesh.faces.shape[0]


def _adaptive_implicit_root_chart_degree(
    domain: AdaptiveImplicitSurface,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    regions: np.ndarray,
    source: AdaptiveImplicitSourceQuery,
    fidelity: SourceFidelityCertificate,
    embedding: GlobalEmbeddingCertificate,
    witness: SourceBoundaryChartWitness,
    /,
) -> tuple[str, int, int]:
    """Recompute complete whole-face charts and their degree one.

    A witness naming every scaffold triangle exactly once keeps the one-to-one
    scaffold argument. Any other target (a remeshed carrier) proves degree one
    from its certified charts: every face is an orientation-preserving local
    homeomorphism onto the zero set and adjacent charts agree on shared edges,
    so the lift is a covering of degree equal to the preimage count of any
    regular value; one certified lifted point per target component has exactly
    one preimage (``_single_sheet_preimages``).
    """
    from ..discretization._cell_geometry_validity import cell_geometry_id
    from .implicit._source_charts import prepare_adaptive_implicit_root_charts

    try:
        root_map = prepare_adaptive_implicit_root_charts(
            mesh,
            geometry,
            source,
            witness,
            tolerance=fidelity.tolerance,
            maximum_normal_deviation=fidelity.maximum_normal_deviation,
        )
    except (
        TypeError,
        ValueError,
        IndexError,
        ArithmeticError,
        FloatingPointError,
    ) as error:
        raise SourceCoverageUnavailable(
            f"Adaptive implicit whole-face root chart is unresolved: {error}",
            status="unresolved",
        ) from error
    evidence = root_map.chart_evidence
    scaffold_mesh = domain.mesh
    if scaffold_mesh is None:
        raise SourceCoverageUnavailable(
            "Adaptive implicit source has no retained scaffold triangles.",
            status="unresolved",
        )
    source_count = int(np.asarray(scaffold_mesh.faces).shape[0])
    scaffold_triangle_cells = np.asarray(
        domain.scaffold.triangle_cell_rows, dtype=np.int64
    )
    scaffold_triangle_edges = np.asarray(
        domain.scaffold.triangle_root_edge_rows, dtype=np.int64
    )
    if scaffold_triangle_cells.shape != (
        source_count,
    ) or scaffold_triangle_edges.shape != (source_count, 3):
        raise SourceCoverageUnavailable(
            "Adaptive implicit source scaffold omits triangle-to-root ancestry.",
            status="violated",
        )
    target_count = len(evidence)
    witness_count = len(witness.source_chart_ids)
    if (
        len(region_ids := tuple(domain.region_ids)) != 1
        or regions.size != sum(block.cell_count for block in mesh.blocks)
        or np.any(regions != 0)
        or source_count == 0
        or target_count != witness_count
    ):
        raise SourceCoverageUnavailable(
            "Adaptive implicit root charts do not give a complete one-to-one material boundary cover.",
            status="violated",
        )
    if (
        root_map.source_id != source.source_id
        or root_map.source_revision != source.source_revision
        or root_map.mesh_id != mesh.mesh_id
        or root_map.topology_id != mesh.topology_id
        or root_map.geometry_id != cell_geometry_id(geometry)
        or not root_map.frame_id
    ):
        raise SourceCoverageUnavailable(
            "Adaptive implicit root chart changes its source, target, or coordinate-frame identity.",
            status="violated",
        )
    witness_cells = np.asarray(witness.implicit_cell_keys, dtype=np.int64)
    witness_edges = np.asarray(witness.implicit_root_edge_keys, dtype=np.int64)
    if witness_cells.shape != (witness_count, 4) or witness_edges.shape != (
        witness_count,
        3,
        2,
        3,
    ):
        raise SourceCoverageUnavailable(
            "Adaptive implicit chart witness omits exact scaffold cell or root-edge ancestry.",
            status="violated",
        )
    chart_rows = []
    for row, chart in enumerate(evidence):
        expected_edges: tuple[
            tuple[_GridPoint, _GridPoint],
            tuple[_GridPoint, _GridPoint],
            tuple[_GridPoint, _GridPoint],
        ] = (
            (
                (
                    int(witness_edges[row, 0, 0, 0]),
                    int(witness_edges[row, 0, 0, 1]),
                    int(witness_edges[row, 0, 0, 2]),
                ),
                (
                    int(witness_edges[row, 0, 1, 0]),
                    int(witness_edges[row, 0, 1, 1]),
                    int(witness_edges[row, 0, 1, 2]),
                ),
            ),
            (
                (
                    int(witness_edges[row, 1, 0, 0]),
                    int(witness_edges[row, 1, 0, 1]),
                    int(witness_edges[row, 1, 0, 2]),
                ),
                (
                    int(witness_edges[row, 1, 1, 0]),
                    int(witness_edges[row, 1, 1, 1]),
                    int(witness_edges[row, 1, 1, 2]),
                ),
            ),
            (
                (
                    int(witness_edges[row, 2, 0, 0]),
                    int(witness_edges[row, 2, 0, 1]),
                    int(witness_edges[row, 2, 0, 2]),
                ),
                (
                    int(witness_edges[row, 2, 1, 0]),
                    int(witness_edges[row, 2, 1, 1]),
                    int(witness_edges[row, 2, 1, 2]),
                ),
            ),
        )
        source_cell_key = (
            int(witness_cells[row, 0]),
            int(witness_cells[row, 1]),
            int(witness_cells[row, 2]),
            int(witness_cells[row, 3]),
        )
        if (
            chart.row != row
            or chart.target_cell_global_id != int(witness.cell_global_ids[row])
            or chart.target_entity_global_id != int(witness.entity_global_ids[row])
            or chart.target_local_entity_index != int(witness.local_entity_indices[row])
            or chart.source_cell_key != source_cell_key
            or chart.root_edge_keys != expected_edges
        ):
            raise SourceCoverageUnavailable(
                "Adaptive implicit chart evidence changes retained target-face or source-root ancestry.",
                status="violated",
            )
        normal_lower, normal_upper = chart.source_normal_deviation_bounds
        if (
            chart.status != "certified"
            or not math.isfinite(chart.field_residual_bound)
            or chart.field_residual_bound < 0.0
            or not math.isfinite(chart.target_to_root_bound)
            or chart.target_to_root_bound < 0.0
            or not math.isfinite(chart.target_to_root_lower_bound)
            or chart.target_to_root_lower_bound < 0.0
            or chart.target_to_root_lower_bound > chart.target_to_root_bound
            or not math.isfinite(normal_lower)
            or not math.isfinite(normal_upper)
            or normal_lower < 0.0
            or normal_upper < normal_lower
            or normal_upper > math.pi
            or not math.isfinite(chart.chart_jacobian_lower)
            or chart.chart_jacobian_lower <= 0.0
            or not math.isfinite(chart.chart_injectivity_margin)
            or chart.chart_injectivity_margin <= 0.0
            or not math.isfinite(chart.source_gradient_lower)
            or chart.source_gradient_lower <= 0.0
        ):
            detail = (
                chart.detail
                or "Adaptive implicit chart lacks finite positive whole-face proof bounds."
            )
            raise SourceCoverageUnavailable(detail, status="unresolved")
        chart_rows.append(
            (
                chart.target_cell_global_id,
                chart.target_entity_global_id,
                chart.target_local_entity_index,
                chart.source_cell_key,
                chart.root_edge_keys,
                chart.chart_jacobian_lower,
                chart.chart_injectivity_margin,
                chart.field_residual_bound,
                chart.target_to_root_lower_bound,
                chart.target_to_root_bound,
                chart.source_normal_deviation_bounds,
            )
        )
    if _names_each_scaffold_triangle_once(domain, witness):
        degree = canonical_fingerprint(
            {
                "kind": "adaptive-implicit-oriented-material-boundary-degree-one",
                "domain": domain.domain_id,
                "source": (source.source_id, source.source_revision, root_map.frame_id),
                "region": region_ids[0],
                "chart_witness": witness.witness_id,
                "root_chart_map": root_map.witness_id,
                "tolerance": fidelity.tolerance,
                "maximum_normal_deviation": fidelity.maximum_normal_deviation,
                "scaffold_triangle_count": source_count,
                "target_chart_count": target_count,
                "target_regions": array_tree_fingerprint(regions),
                "embedding": embedding.certificate_id,
                "charts": tuple(chart_rows),
                "degree": 1,
            }
        )
        return degree, source_count, target_count
    preimages = _single_sheet_preimages(mesh, geometry, regions, root_map, witness)
    degree = canonical_fingerprint(
        {
            "kind": "adaptive-implicit-oriented-material-boundary-covering-degree-one",
            "domain": domain.domain_id,
            "source": (source.source_id, source.source_revision, root_map.frame_id),
            "region": region_ids[0],
            "chart_witness": witness.witness_id,
            "root_chart_map": root_map.witness_id,
            "tolerance": fidelity.tolerance,
            "maximum_normal_deviation": fidelity.maximum_normal_deviation,
            "scaffold_triangle_count": source_count,
            "target_chart_count": target_count,
            "target_regions": array_tree_fingerprint(regions),
            "embedding": embedding.certificate_id,
            "charts": tuple(chart_rows),
            "single_preimages": preimages,
            "degree": 1,
        }
    )
    return degree, source_count, target_count


def _names_each_scaffold_triangle_once(
    domain: AdaptiveImplicitSurface, witness: SourceBoundaryChartWitness, /
) -> bool:
    """Whether the witness rows name every scaffold triangle exactly once."""
    scaffold = domain.scaffold
    cells = np.asarray(scaffold.cell_indices, dtype=np.int64)
    root_keys = np.asarray(scaffold.root_edge_ids, dtype=np.int64)

    def key(cell: np.ndarray, edges: np.ndarray, /) -> tuple[object, ...]:
        canonical = sorted(
            tuple(sorted(tuple(int(value) for value in endpoint) for endpoint in edge))
            for edge in edges.reshape((3, 2, 3))
        )
        return (tuple(int(value) for value in cell), tuple(canonical))

    scaffold_keys = [
        key(cells[int(cell_row)], root_keys[np.asarray(edge_rows, dtype=np.int64)])
        for cell_row, edge_rows in zip(
            np.asarray(scaffold.triangle_cell_rows, dtype=np.int64),
            np.asarray(scaffold.triangle_root_edge_rows, dtype=np.int64),
            strict=True,
        )
    ]
    witness_cells = np.asarray(witness.implicit_cell_keys, dtype=np.int64)
    witness_edges = np.asarray(witness.implicit_root_edge_keys, dtype=np.int64)
    witness_keys = [
        key(witness_cells[row], witness_edges[row])
        for row in range(witness_cells.shape[0])
    ]
    return (
        len(witness_keys) == len(scaffold_keys)
        and len(set(witness_keys)) == len(witness_keys)
        and set(witness_keys) == set(scaffold_keys)
    )


# Relative rounding slack of the binary64 face-distance lower bound.
_DISTANCE_SLACK = 2.0**-30


def _face_distance_lower(point: np.ndarray, triangle: np.ndarray, /) -> float:
    """Lower bound on dist(point, triangle) from its plane and edge half-planes.

    The triangle lies in its plane P and in each in-plane half-plane H_k behind
    edge k, so dist^2 >= dist(point, P)^2 + max_k dist(point, H_k)^2 (the edge
    normals are orthogonal to the plane normal). Rounding is absorbed by a
    relative slack.
    """
    first, second, third = triangle
    normal = np.cross(second - first, third - first)
    length = float(np.linalg.norm(normal))
    if not length > 0.0:
        return 0.0
    normal = normal / length
    plane = abs(float(np.dot(point - first, normal)))
    outside = 0.0
    for start, end in ((first, second), (second, third), (third, first)):
        edge_normal = np.cross(end - start, normal)
        edge_length = float(np.linalg.norm(edge_normal))
        if edge_length > 0.0:
            outside = max(
                outside, float(np.dot(point - start, edge_normal)) / edge_length
            )
    scale = float(np.max(np.abs(triangle))) + float(np.max(np.abs(point)))
    return max(0.0, math.hypot(plane, outside) - _DISTANCE_SLACK * scale)


def _single_sheet_preimages(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    regions: np.ndarray,
    root_map: AdaptiveImplicitRootChartMap,
    witness: SourceBoundaryChartWitness,
    /,
) -> tuple[tuple[int, tuple[float, float, float], float], ...]:
    """Certify exactly one preimage of one lifted point per target component.

    Per connected target component, the face with the largest inradius lifts
    its reference centroid to a certified zero-set point q (within its root
    error). Every lift moves a face point by at most the face's certified
    target-to-root bound, so a face whose distance lower bound to q exceeds
    that bound plus q's error has no preimage of q; the lifting face itself
    is injective. A face the prepared bound cannot exclude is re-proved by
    splitting only its pieces whose lifted-point boxes still reach q, until each
    is excluded, lifts into q's error, or cannot be split further in binary64.
    A face that still reaches q (a folded or duplicated sheet) leaves the degree
    unresolved.
    """
    from ..discretization._coordinate_enclosure import affine_map
    from ._domain_numerics import cell_coordinate_expressions

    if not all(
        expressions is not None and affine_map(expressions, 3) is not None
        for expressions in cell_coordinate_expressions(mesh, geometry)
    ):
        raise SourceCoverageUnavailable(
            "A non-scaffold implicit boundary degree requires affine target faces.",
            status="unresolved",
        )
    face_owners, face_by_cell, _signs = _volume_face_owners(mesh, regions)
    vertex_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        )
    }
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    keys: list[tuple[int, ...]] = []
    for cell_id, local_face in zip(
        witness.cell_global_ids, witness.local_entity_indices, strict=True
    ):
        face_key = face_by_cell.get((int(cell_id), int(local_face)))
        if face_key is None or len(face_key) != 3:
            raise SourceCoverageUnavailable(
                "Implicit witness names a target face outside the volume boundary.",
                status="violated",
            )
        keys.append(face_key)
    triangles = coordinates[
        np.asarray(
            [[vertex_rows[vertex] for vertex in key] for key in keys], dtype=np.int64
        )
    ]
    required = {
        key
        for key, adjacent in face_owners.items()
        if len(adjacent) == 1 or adjacent[0][1] != adjacent[1][1]
    }
    if len(set(keys)) != len(keys) or set(keys) != required:
        raise SourceCoverageUnavailable(
            "Implicit witness repeats or omits a target boundary facet.",
            status="violated",
        )
    parent = list(range(len(keys)))

    def root(row: int, /) -> int:
        while parent[row] != row:
            parent[row] = parent[parent[row]]
            row = parent[row]
        return row

    edge_owner: dict[tuple[int, int], int] = {}
    for row, key in enumerate(keys):
        for edge in ((key[0], key[1]), (key[1], key[2]), (key[0], key[2])):
            other = edge_owner.setdefault(edge, row)
            parent[root(other)] = root(row)
    sides = np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2)
    twice_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    inradius = twice_area / np.sum(sides, axis=1)
    bounds = tuple(chart.target_to_root_bound for chart in root_map.chart_evidence)
    components: dict[int, int] = {}
    for row in range(len(keys)):
        component = root(row)
        best = components.get(component)
        if best is None or inradius[row] > inradius[best]:
            components[component] = row
    preimages: list[tuple[int, tuple[float, float, float], float]] = []
    for lifting_row in sorted(components.values()):
        corners = np.asarray(witness.target_reference_vertices[lifting_row])
        centroid = tuple(
            sum((Fraction(float(value)) for value in corners[:, axis]), Fraction(0)) / 3
            for axis in range(corners.shape[1])
        )
        lifted = root_map.project_target(lifting_row, centroid)
        point = np.asarray(lifted.point, dtype=np.float64)
        if not np.all(np.isfinite(point)) or not math.isfinite(lifted.root_error):
            raise SourceCoverageUnavailable(
                "An implicit target component has no certified lifted point.",
                status="unresolved",
            )
        distances = {
            row: _face_distance_lower(point, triangles[row])
            for row in range(len(keys))
            if row != lifting_row
        }
        blocking = [
            row
            for row, distance in distances.items()
            if distance <= math.nextafter(bounds[row] + lifted.root_error, math.inf)
        ]
        excluded = (
            root_map.excludes_point(blocking, point, lifted.root_error)
            if blocking
            else ()
        )
        covering = tuple(
            row for row, decided in zip(blocking, excluded, strict=True) if not decided
        )
        if covering:
            raise SourceCoverageUnavailable(
                "A second implicit target sheet may cover the zero-set point lifted "
                f"from target face row {lifting_row}: target face rows {covering} "
                "still reach it within its certified error or at binary64 "
                "resolution; the boundary degree is not one.",
                status="unresolved",
            )
        preimages.append(
            (
                lifting_row,
                (float(point[0]), float(point[1]), float(point[2])),
                float(lifted.root_error),
            )
        )
    return tuple(preimages)


def _cross3(points: _Triangle3, /) -> _Point3:
    first = (
        points[1][0] - points[0][0],
        points[1][1] - points[0][1],
        points[1][2] - points[0][2],
    )
    second = (
        points[2][0] - points[0][0],
        points[2][1] - points[0][1],
        points[2][2] - points[0][2],
    )
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot3(
    first: tuple[Fraction, Fraction, Fraction],
    second: tuple[Fraction, Fraction, Fraction],
    /,
) -> Fraction:
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def _target_face_chart_keys(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> dict[tuple[int, int], _Triangle3]:
    from ..discretization._reference_cell import reference_cell_topology
    from ._domain_numerics import cell_coordinate_expressions, evaluate_map

    maps = cell_coordinate_expressions(mesh, geometry)
    result: dict[tuple[int, int], _Triangle3] = {}
    cell_row = 0
    for block in mesh.blocks:
        if block.cell_kind in ("polygon", "polyhedron"):
            raise SourceCoverageUnavailable(
                "Adaptive implicit root ancestry requires standard reference volume cells.",
                status="unresolved",
            )
        reference = reference_cell_topology(block.cell_kind)
        if reference.dimension != 3:
            raise SourceCoverageUnavailable(
                "Implicit boundary ancestry requires volume cells.", status="unresolved"
            )
        vertices = tuple(
            tuple(_fraction(float(value)) for value in point)
            for point in reference.vertices
        )
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        for row in range(block.cell_count):
            cell_id = int(identifiers[row])
            expressions = maps[cell_row]
            if expressions is None:
                raise SourceCoverageUnavailable(
                    "Implicit target face has no exact represented coordinate map.",
                    status="unresolved",
                )
            for local_face, face in enumerate(reference.entities[2]):
                points = tuple(
                    _point3(tuple(evaluate_map(expressions, vertices[int(slot)])))
                    for slot in face
                )
                if len(points) != 3:
                    raise SourceCoverageUnavailable(
                        "Implicit root charts currently require triangular target faces.",
                        status="unresolved",
                    )
                triangle = (points[0], points[1], points[2])
                ordered = sorted(triangle)
                key = (cell_id, local_face)
                if key in result:
                    raise SourceCoverageUnavailable(
                        "Target cell identities are not unique.", status="violated"
                    )
                result[key] = (ordered[0], ordered[1], ordered[2])
            cell_row += 1
    if cell_row != len(maps):
        raise SourceCoverageUnavailable(
            "Target coordinate maps do not align with volume cells.", status="violated"
        )
    return result


def _target_facets(
    mesh: CellMesh, geometry: CellGeometrySpec, cell_regions: np.ndarray, /
) -> tuple[tuple[OrientedFacet, ...], tuple[str, ...]]:
    from ._domain_numerics import cell_coordinate_expressions
    from ._domain_planar_boundary import oriented_target_facets

    try:
        maps = cell_coordinate_expressions(mesh, geometry)
        facets, issues = oriented_target_facets(
            mesh, cell_regions, maps, geometry=geometry
        )
    except (TypeError, ValueError, IndexError, ArithmeticError) as error:
        return (), (f"Target boundary extraction failed: {error}",)
    return facets, tuple(issue.detail for issue in issues)


def approximate_source_coverage(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: MeshingDomain | AdaptiveImplicitSurface,
    cell_regions: np.ndarray,
    /,
    *,
    source: SourceBoundaryQuery | None,
    fidelity: SourceFidelityCertificate | None,
    embedding: GlobalEmbeddingCertificate,
) -> SourceCoverageAttempt:
    """Prove bounded oriented material coverage from source-chart homotopy data.

    The fidelity certificate must be current for these operands; its proof
    validated its boundary-chart witness on them, so the witness is not
    validated again.
    """

    def fail(
        check: str,
        detail: str,
        status: Literal["violated", "unresolved"] = "unresolved",
        entity_ids: tuple[int, ...] = (),
    ) -> SourceCoverageAttempt:
        return SourceCoverageAttempt(
            None, (SourceCoverageFailure(check, status, detail, entity_ids),)
        )

    if not isinstance(domain, (MeshingDomain, AdaptiveImplicitSurface)):
        return fail(
            "source_boundary_domain",
            "No source-authoritative bounded coverage proof is implemented for this domain type.",
        )

    if source is None or fidelity is None:
        return fail(
            "source_boundary_homotopy",
            "Bounded source coverage requires both the actual source query and its current continuous fidelity certificate.",
        )
    if not isinstance(source, SourceBoundaryQuery) or not isinstance(
        fidelity, SourceFidelityCertificate
    ):
        return fail(
            "source_boundary_homotopy",
            "Source coverage premises have the wrong runtime types.",
            "violated",
        )
    if not isinstance(embedding, GlobalEmbeddingCertificate):
        return fail(
            "source_boundary_embedding",
            "A current global embedding certificate is required.",
            "violated",
        )
    if embedding.status != "certified":
        return fail(
            "source_boundary_embedding",
            "A current certified global embedding is required.",
            "violated" if embedding.status == "violated" else "unresolved",
        )
    try:
        embedding.binding.require(mesh, geometry)
        fidelity.require_current(mesh, geometry, source)
    except (TypeError, ValueError, ArithmeticError) as error:
        return fail(
            "source_boundary_premise",
            f"Source or embedding premises do not revalidate: {error}",
            "violated",
        )
    if fidelity.status != "certified":
        return fail(
            "source_boundary_fidelity",
            "A certified whole-boundary two-directed fidelity certificate is required.",
        )
    if (fidelity.binding.source_id, fidelity.binding.source_revision) != (
        source.source_id,
        source.source_revision,
    ):
        return fail(
            "source_boundary_identity",
            "Fidelity and source query bind different source revisions.",
            "violated",
        )
    if not math.isfinite(
        max(fidelity.mesh_to_source_upper, fidelity.source_to_mesh_upper)
    ):
        return fail(
            "source_boundary_distance",
            "Source fidelity does not provide finite directed boundary bounds.",
        )

    adaptive_query: AdaptiveImplicitSourceQuery | None = None
    source_frame_id: str | None = None
    cover = fidelity.coverage
    if isinstance(domain, AdaptiveImplicitSurface):
        from .implicit._source_charts import (
            _AdaptiveImplicitBoundaryCarrier,
            _source_authority,
        )

        if not (
            isinstance(source, AdaptiveImplicitBoundarySource)
            or isinstance(source, _AdaptiveImplicitBoundaryCarrier)
        ):
            return fail(
                "implicit_source_identity",
                "Adaptive implicit coverage requires its native adaptive source query.",
                "violated",
            )
        adaptive_query = source
        try:
            adaptive_source, source_frame_id = _source_authority(source)
        except (AttributeError, TypeError, ValueError) as error:
            return fail("implicit_source_frame", str(error))
        expected_source = domain.boundary_source
        from ._mesh_certificates import _same_tree_content

        # A native carrier republishes the discovery under its own revision; it
        # binds the domain through the discovery's adaptive boundary query.
        if (
            (adaptive_source.source_id, adaptive_source.source_revision)
            != (expected_source.source_id, expected_source.source_revision)
            or not _same_tree_content(adaptive_source.surface, domain)
            or source.source_id != adaptive_source.source_id
        ):
            return fail(
                "implicit_source_identity",
                "Native implicit query does not bind the current adaptive source and domain.",
                "violated",
            )
        if cover is not None and (
            (cover.source_id, cover.source_revision)
            not in {
                (source.source_id, source.source_revision),
                (adaptive_source.source_id, adaptive_source.source_revision),
            }
            or not cover.complete
            or cover.semantics != "certified"
            or cover.findings
        ):
            return fail(
                "implicit_source_cover",
                "An optional implicit chart cover conflicts with its revalidated source revision.",
                "violated",
            )
    elif (
        cover is None
        or not cover.complete
        or cover.semantics != "certified"
        or cover.findings
        or (cover.source_id, cover.source_revision)
        != (source.source_id, source.source_revision)
    ):
        return fail(
            "source_boundary_cover",
            "The revalidated complete source chart cover is not certified.",
        )
    witness = fidelity.witness
    if not isinstance(witness, SourceBoundaryChartWitness):
        return fail(
            "source_boundary_chart_witness",
            "Bounded source coverage requires a current complete boundary-chart witness.",
        )
    if (
        witness.source_id != source.source_id
        or witness.source_revision != source.source_revision
    ):
        return fail(
            "source_boundary_chart_identity",
            "Boundary chart witness binds another source revision.",
            "violated",
        )
    regions = np.asarray(cell_regions)
    if isinstance(domain, MeshingDomain):
        region_ids = tuple(region.name for region in domain.regions)
        domain_id = domain.domain_id
    else:
        if not isinstance(domain, AdaptiveImplicitSurface):
            return fail(
                "source_boundary_domain",
                "No source-authoritative bounded coverage proof is implemented for this domain type.",
            )
        region_ids = domain.region_ids
        domain_id = domain.domain_id
    from ._domain_certification import coverage_region_occurrence_paths

    region_occurrence_paths = coverage_region_occurrence_paths(domain)
    if (
        regions.ndim != 1
        or regions.dtype.kind not in "iu"
        or regions.size != sum(block.cell_count for block in mesh.blocks)
    ):
        return fail(
            "source_boundary_region_assignment",
            "Dense material assignments do not align with target volume cells.",
            "violated",
        )
    if not region_ids or np.any(regions < 0) or np.any(regions >= len(region_ids)):
        return fail(
            "source_boundary_region_assignment",
            "Every target volume cell must have an explicit source material region.",
            "unresolved",
        )
    try:
        if isinstance(domain, MeshingDomain):
            if (source.source_id, source.source_revision) != (
                domain.source_id,
                domain.source_revision,
            ):
                return fail(
                    "source_boundary_identity",
                    "CAD source query does not bind the retained MeshingDomain.",
                    "violated",
                )
            source_count, degrees = _cad_chart_partition(
                domain, source, witness, mesh, regions
            )
            target_count = witness.cell_global_ids.size
        else:
            if not isinstance(domain, AdaptiveImplicitSurface):
                return fail(
                    "source_boundary_domain",
                    "No source-authoritative bounded coverage proof is implemented for this domain type.",
                )
            if adaptive_query is None:
                return fail(
                    "implicit_source_identity",
                    "Adaptive root coverage has no narrowed native source query.",
                    "violated",
                )
            if (
                source.source_id != domain.source_id
                or region_ids != domain.region_ids
                or len(region_ids) != 1
                or np.any(regions != 0)
            ):
                return fail(
                    "implicit_material_identity",
                    "Implicit source material identity or per-cell assignments differ from its one declared region.",
                    "violated",
                )
            if witness.chart_dimension != 2 or witness.reference_dimension != 3:
                return fail(
                    "implicit_chart_dimension",
                    "Implicit volume chart witness has the wrong dimensions.",
                )
            scaffold_mesh = domain.mesh
            if scaffold_mesh is None:
                return fail(
                    "implicit_source_scaffold",
                    "Adaptive implicit source has no retained scaffold triangles.",
                )
            source_count = int(np.asarray(scaffold_mesh.faces).shape[0])
            if source_count == 0:
                return fail(
                    "implicit_source_scaffold",
                    "Adaptive implicit source scaffold is empty.",
                )
            try:
                degree, source_count, target_count = _adaptive_implicit_root_chart_degree(
                    domain,
                    mesh,
                    geometry,
                    regions,
                    adaptive_query,
                    fidelity,
                    embedding,
                    witness,
                )
            except SourceCoverageUnavailable as chart_error:
                if chart_error.status != "unresolved":
                    raise
                frame_id = source_frame_id
                if not frame_id:
                    raise chart_error
                from ..discretization._coordinate_enclosure import affine_map
                from ._domain_numerics import cell_coordinate_expressions

                try:
                    target_maps = cell_coordinate_expressions(mesh, geometry)
                    exact_affine = all(
                        expressions is not None and affine_map(expressions, 3) is not None
                        for expressions in target_maps
                    )
                except (TypeError, ValueError, IndexError, ArithmeticError):
                    exact_affine = False
                if not exact_affine:
                    raise chart_error
                target_facets, target_issues = _target_facets(mesh, geometry, regions)
                if target_issues:
                    raise chart_error
                source_triangles = np.asarray(scaffold_mesh.vertices)[
                    np.asarray(scaffold_mesh.faces, dtype=np.int64)
                ]
                source_keys = {
                    tuple(
                        sorted(
                            tuple(_fraction(float(value)) for value in point)
                            for point in triangle
                        )
                    )
                    for triangle in source_triangles
                }
                target_keys = tuple(
                    tuple(sorted(tuple(point) for point in facet.vertices))
                    for facet in target_facets
                )
                if (
                    len(target_keys) != source_count
                    or len(set(target_keys)) != len(target_keys)
                    or len(source_keys) != source_count
                    or set(target_keys) != source_keys
                ):
                    raise chart_error
                try:
                    degree, source_count = _implicit_root_derivation(
                        domain, mesh, geometry, regions, witness
                    )
                except SourceCoverageUnavailable as exact_error:
                    if exact_error.status == "violated":
                        raise
                    raise chart_error from exact_error
                target_count = witness.cell_global_ids.size
                if target_count != source_count:
                    return fail(
                        "implicit_chart_coverage",
                        "Implicit witness rows do not cover every root scaffold triangle.",
                        "violated",
                    )
                degree = canonical_fingerprint(
                    {
                        "kind": "frame-bound-exact-implicit-degree",
                        "base_degree": degree,
                        "frame_id": frame_id,
                    }
                )
            degrees = (degree,)
    except SourceCoverageUnavailable as error:
        return fail("source_boundary_homotopy", str(error), error.status)
    except (
        TypeError,
        ValueError,
        IndexError,
        ArithmeticError,
        FloatingPointError,
    ) as error:
        return fail(
            "source_boundary_homotopy", f"Source boundary homotopy is unresolved: {error}"
        )
    distance = max(fidelity.mesh_to_source_upper, fidelity.source_to_mesh_upper)
    witness_id = canonical_fingerprint(
        {
            "kind": "bounded-source-domain-coverage",
            "domain": domain_id,
            "fidelity": fidelity.certificate_id,
            "embedding": embedding.certificate_id,
            "chart_witness": witness.witness_id,
            "source_frame": source_frame_id,
            "region_ids": region_ids,
            "region_occurrence_paths": region_occurrence_paths,
            "cell_regions": array_tree_fingerprint(regions),
            "boundary_degrees": degrees,
            "distance_upper": repr(distance),
        }
    )
    bounded = BoundedSourceCoverage(
        domain_id,
        source.source_id,
        source.source_revision,
        region_ids,
        region_occurrence_paths,
        int(source_count),
        int(target_count),
        int(source_count),
        tuple(degrees),
        float(distance),
        witness_id,
    )
    return SourceCoverageAttempt(bounded, ())


__all__ = [
    "BoundedSourceCoverage",
    "ExactCadSourceBoundary",
    "SourceCoverageAttempt",
    "SourceCoverageFailure",
    "SourceCoverageUnavailable",
    "approximate_source_coverage",
    "cad_material_sides",
    "exact_planar_cad_boundary",
]
