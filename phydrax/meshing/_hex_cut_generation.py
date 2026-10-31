# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Conforming integer-grid hexes for exact orthogonal PLC partitions."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from typing import Any, cast, final, TypeAlias

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._geometry_predicates import orient3d, PredicateMode, PredicateSign
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..discretization._cell_complex import TetrahedralConnectivity
from ..discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    coordinate_effective_basis,
    PerCellRestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityPolicy,
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from ..discretization._cell_mesh import PolyhedralBlock, SimplicialConnectivity
from ..discretization._coordinate_enclosure import (
    add,
    axes,
    bernstein_coefficients,
    compose_expression,
    constant,
    derivative,
    evaluate_expression,
    ExactScalar,
    Expression,
    expression_equal,
    linear_combinations,
    multiply,
    Polynomial,
    RationalPolynomial,
    scale,
    sum_polynomials,
)
from ..discretization._exact_algebraic_geometry import RadicalElement
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._nested_reference_types import ReferenceCellMapWitnesses
from ..discretization._reference_cell import reference_cell_topology
from ..geometry._exact_polyhedral_geometry import projected_convex_hull
from ..geometry._global_embedding import (
    certify_global_embedding,
    GlobalEmbeddingCertificate,
)
from ..linalg._exact import solve_exact
from ..meshing._quality import evaluate_cell_quality
from ..typing import checked, ConvertibleToArray
from ._dual_extraction import fresh_vertex_global_ids
from ._hex_cut_preparation import (
    BalancedCutGridPlan,
    OrthogonalPlcGridPlan,
)
from ._hex_generation import _compose_restricted_dual_chart
from ._hex_schedule import NativeHexGridSchedule
from ._plc_source import PiecewiseLinearComplex


Point3: TypeAlias = tuple[Fraction, Fraction, Fraction]
ExactPoint3: TypeAlias = tuple[ExactScalar, ExactScalar, ExactScalar]
Weights: TypeAlias = tuple[tuple[int, Fraction], ...]


def _fraction_point3(values: np.ndarray, /) -> Point3:
    return (
        Fraction.from_float(float(values[0])),
        Fraction.from_float(float(values[1])),
        Fraction.from_float(float(values[2])),
    )


def _require_global_embedding(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validity: CellValidityCertificate,
    description: str,
    /,
) -> GlobalEmbeddingCertificate:
    embedding = certify_global_embedding(mesh, geometry, validity)
    if embedding.status == "certified":
        return embedding
    details = tuple(
        f"{finding.check} ({finding.entity_kind} {finding.entity_ids}): {finding.detail}"
        for finding in embedding.findings
    )
    obstruction = (
        "; ".join(details) if details else "no exact cell-contact witness was produced"
    )
    raise ValueError(f"{description} has no global embedding proof: {obstruction}.")


def _require_tetrahedral_global_embedding(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validity: CellValidityCertificate,
    /,
) -> GlobalEmbeddingCertificate:
    return _require_global_embedding(mesh, geometry, validity, "Source tetrahedral fill")


def _exact_source_coordinate_bank(
    values: Sequence[Sequence[object]],
    /,
) -> tuple[tuple[ExactScalar, ...], ...]:
    rows: list[tuple[ExactScalar, ...]] = []
    for row in values:
        exact_row: list[ExactScalar] = []
        for value in row:
            if isinstance(value, (Fraction, RadicalElement)):
                exact_row.append(value)
            elif isinstance(value, (int, np.integer)) and not isinstance(
                value, (bool, np.bool_)
            ):
                exact_row.append(Fraction(int(value)))
            elif isinstance(value, (float, np.floating)):
                exact_row.append(Fraction.from_float(float(value)))
            else:
                raise TypeError(
                    "Exact cut charts require rational or real-algebraic source coordinates."
                )
        rows.append(tuple(exact_row))
    return tuple(rows)


def _exact_scalar_key(value: ExactScalar, /) -> tuple[tuple[int, int, int], ...]:
    terms = (
        value.terms
        if isinstance(value, RadicalElement)
        else (() if not value else ((1, value),))
    )
    return tuple(
        (int(radicand), coefficient.numerator, coefficient.denominator)
        for radicand, coefficient in terms
    )


def _require_fraction(
    value: ExactScalar,
    context: str,
    /,
) -> Fraction:
    if isinstance(value, Fraction):
        return value
    raise ValueError(context)


def _predicate_sign(*points: np.ndarray) -> int:
    result = orient3d(*points, mode=PredicateMode.EXACT)
    if not bool(np.all(np.asarray(result.certain))):
        raise ArithmeticError(
            "Exact PLC grid classification returned an uncertain orientation."
        )
    return int(np.asarray(result.signs).item())


def _point_in_tetrahedron(point: np.ndarray, corners: np.ndarray, /) -> bool:
    a, b, c, d = corners
    orientation = _predicate_sign(a, b, c, d)
    if orientation == int(PredicateSign.ZERO):
        raise ValueError(
            "Tetrahedral source classification encountered a degenerate cell."
        )
    signs = (
        _predicate_sign(point, b, c, d),
        _predicate_sign(a, point, c, d),
        _predicate_sign(a, b, point, d),
        _predicate_sign(a, b, c, point),
    )
    return all(sign == 0 or sign == orientation for sign in signs)


@final
class OrthogonalPlcHexConstruction(StrictModule, NonTrainableState):
    """All-hexahedral candidate over a fully grid-aligned PLC partition."""

    mesh: CellMesh
    geometry: CellGeometrySpec
    validity: CellValidityCertificate
    global_embedding: GlobalEmbeddingCertificate
    cell_regions: np.ndarray
    integer_cells: np.ndarray
    scaled_jacobian_lower: np.ndarray
    mean_ratio_lower: np.ndarray
    aspect_ratio_upper: np.ndarray
    source_mesh_id: str = eqx.field(static=True)
    source_complex_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    construction_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        validity: CellValidityCertificate,
        global_embedding: GlobalEmbeddingCertificate,
        cell_regions: ConvertibleToArray,
        integer_cells: ConvertibleToArray,
        scaled_jacobian_lower: ConvertibleToArray,
        mean_ratio_lower: ConvertibleToArray,
        aspect_ratio_upper: ConvertibleToArray,
        source_mesh_id: str,
        source_complex_id: str,
        plan_id: str,
        construction_id: str,
        /,
    ) -> None:
        regions = np.asarray(cell_regions, dtype=np.int64)
        integer = np.asarray(integer_cells, dtype=np.int64)
        scaled = np.asarray(scaled_jacobian_lower, dtype=np.float64)
        mean = np.asarray(mean_ratio_lower, dtype=np.float64)
        if not isinstance(global_embedding, GlobalEmbeddingCertificate):
            raise TypeError("global_embedding must be a GlobalEmbeddingCertificate.")
        if (
            global_embedding.status != "certified"
            or global_embedding.validity.certificate_id != validity.certificate_id
        ):
            raise ValueError(
                "Orthogonal PLC global embedding must bind its current validity."
            )
        global_embedding.binding.require(mesh, geometry)
        aspect = np.asarray(aspect_ratio_upper, dtype=np.float64)
        connectivity = mesh.connectivity
        if not isinstance(connectivity, HexahedralConnectivity):
            raise ValueError(
                "Orthogonal PLC construction requires hexahedral connectivity."
            )
        cells = connectivity.cell_count
        if mesh.topological_dimension != 3 or any(
            block.cell_kind != "hexahedron" for block in mesh.blocks
        ):
            raise ValueError(
                "Orthogonal PLC grid construction must contain only hexahedra."
            )
        if regions.shape != (cells,) or integer.shape != (cells, 3):
            raise ValueError(
                "Orthogonal grid region and integer-cell rows must match cells."
            )
        if any(value.shape != (cells,) for value in (scaled, mean, aspect)):
            raise ValueError("Orthogonal grid quality evidence must match cells.")
        if not all(np.all(np.isfinite(value)) for value in (scaled, mean, aspect)):
            raise ValueError("Orthogonal grid quality evidence must be finite.")
        if not all((source_mesh_id, source_complex_id, plan_id, construction_id)):
            raise ValueError("Orthogonal grid construction identities must be non-empty.")
        self.mesh = mesh
        self.geometry = geometry
        self.validity = validity
        self.global_embedding = global_embedding
        self.cell_regions = regions
        self.integer_cells = integer
        self.scaled_jacobian_lower = scaled
        self.mean_ratio_lower = mean
        self.aspect_ratio_upper = aspect
        self.source_mesh_id = source_mesh_id
        self.source_complex_id = source_complex_id
        self.plan_id = plan_id
        self.construction_id = construction_id


def _candidate_grid_bins(
    mesh: CellMesh,
    plan: OrthogonalPlcGridPlan,
    /,
) -> tuple[dict[tuple[int, int, int], list[int]], tuple[tuple[Fraction, ...], ...]]:
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    exact_points = tuple(_fraction_point3(point) for point in coordinates)
    centers = tuple(
        tuple(
            (first + second) / 2
            for first, second in zip(axis[:-1], axis[1:], strict=True)
        )
        for axis in plan.axis_knots
    )
    bins: dict[tuple[int, int, int], list[int]] = {}
    for cell_row, cell in enumerate(cells.tolist()):
        corners = tuple(exact_points[int(vertex)] for vertex in cell)
        ranges = []
        for axis in range(3):
            lower = min(point[axis] for point in corners)
            upper = max(point[axis] for point in corners)
            first = bisect_left(centers[axis], lower)
            stop = bisect_right(centers[axis], upper)
            ranges.append(range(first, stop))
        for raw_index in product(*ranges):
            index = (int(raw_index[0]), int(raw_index[1]), int(raw_index[2]))
            bins.setdefault(index, []).append(cell_row)
    return bins, centers


def _grid_point(
    index: tuple[Fraction, Fraction, Fraction],
    /,
) -> np.ndarray:
    values = np.asarray([float(value) for value in index], dtype=np.float64)
    for exact, rounded in zip(index, values, strict=True):
        if Fraction.from_float(float(rounded)) != exact:
            raise ValueError(
                "Grid evaluation points must be exactly representable in binary64."
            )
    return values


@checked
def generate_orthogonal_plc_hexes(
    source: PiecewiseLinearComplex,
    source_mesh: CellMesh,
    cell_regions: ConvertibleToArray,
    plan: OrthogonalPlcGridPlan,
    schedule: NativeHexGridSchedule,
    /,
) -> OrthogonalPlcHexConstruction:
    """Construct a conforming grid for an exact axis-aligned PLC partition.

    Grid alignment is a mathematical admission premise: every polygon edge
    follows lattice lines, every source facet lies on a lattice plane, and the
    source tetrahedral fill classifies each open grid cell. Partial cells are
    refused rather than center-sampled, clipped approximately, or replaced by
    dual tetrahedra. The general cut-cell and frame-field routes remain distinct.
    """
    if plan.source_complex_id != source.complex_id:
        raise ValueError("Orthogonal grid plan must bind the exact PLC source complex.")
    if plan.schedule_id != schedule.schedule_id or schedule.route != "balanced_grid":
        raise ValueError("Orthogonal grid plan and balanced-grid schedule must match.")
    if source.geometry is not None or source.boundary != "conforming":
        raise ValueError(
            "Orthogonal grid requires the admitted conforming piecewise-linear PLC."
        )
    if (
        source_mesh.topological_dimension != 3
        or source_mesh.ambient_dimension != 3
        or not source_mesh.blocks
        or any(not _is_tetrahedron_block(block) for block in source_mesh.blocks)
    ):
        raise ValueError(
            "Orthogonal PLC classification requires a tetrahedral source fill."
        )
    source_cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in source_mesh.blocks]
    )
    regions = np.asarray(cell_regions)
    if regions.dtype.kind not in "iu" or regions.shape != (source_cells.shape[0],):
        raise ValueError(
            "cell_regions must contain one integer source region per tetrahedron."
        )
    region_rows = np.asarray(regions, dtype=np.int64)
    if np.any((region_rows < 0) | (region_rows >= len(source.region_ids))):
        raise ValueError("cell_regions must index the PLC's declared regions.")
    source_geometry = CellGeometrySpec.affine(source_mesh)
    source_validity = certify_cell_geometry_validity(source_geometry, mesh=source_mesh)
    if source_validity.invalid_count or source_validity.unresolved_count:
        raise ValueError(
            "Orthogonal PLC source fill must have certified positive tetrahedra."
        )
    _require_tetrahedral_global_embedding(source_mesh, source_geometry, source_validity)

    source_points = np.asarray(source_mesh.coordinates, dtype=np.float64)
    candidate_bins, axis_centers = _candidate_grid_bins(source_mesh, plan)
    occupied: list[tuple[tuple[int, int, int], int]] = []
    for index in sorted(candidate_bins):
        center = _grid_point(
            (
                axis_centers[0][index[0]],
                axis_centers[1][index[1]],
                axis_centers[2][index[2]],
            )
        )
        hits = [
            row
            for row in candidate_bins[index]
            if _point_in_tetrahedron(center, source_points[source_cells[row]])
        ]
        if not hits:
            continue
        labels = {int(region_rows[row]) for row in hits}
        if len(labels) != 1:
            raise ValueError(
                "A grid cell intersects a material interface away from its lattice faces."
            )
        occupied.append((index, labels.pop()))
    if not occupied:
        raise ValueError("The orthogonal grid contains no positive-volume source cells.")

    reference_vertices = np.asarray(
        reference_cell_topology("hexahedron").vertices, dtype=np.int64
    )
    vertex_rows: dict[tuple[int, int, int], int] = {}
    vertex_indices: list[tuple[int, int, int]] = []
    cells: list[list[int]] = []
    output_regions: list[int] = []
    quality_scaled: list[float] = []
    quality_mean: list[float] = []
    quality_aspect: list[float] = []
    for index, region in occupied:
        corners = []
        for offset in reference_vertices:
            address = (
                index[0] + int(offset[0]),
                index[1] + int(offset[1]),
                index[2] + int(offset[2]),
            )
            if address not in vertex_rows:
                vertex_rows[address] = len(vertex_indices)
                vertex_indices.append(address)
            corners.append(vertex_rows[address])
        cells.append(corners)
        output_regions.append(region)
        lengths = np.asarray(
            [
                float(
                    plan.axis_knots[axis][index[axis] + 1]
                    - plan.axis_knots[axis][index[axis]]
                )
                for axis in range(3)
            ],
            dtype=np.float64,
        )
        aspect = float(np.max(lengths) / np.min(lengths))
        mean = float(3.0 * np.prod(lengths) ** (2.0 / 3.0) / np.sum(lengths * lengths))
        if (
            1.0 < schedule.minimum_scaled_jacobian
            or mean < schedule.minimum_mean_ratio
            or aspect > schedule.maximum_aspect_ratio
        ):
            raise ValueError(
                "Orthogonal grid cell fails the declared whole-cell quality schedule."
            )
        quality_scaled.append(1.0)
        quality_mean.append(mean)
        quality_aspect.append(aspect)

    coordinates = np.asarray(
        [
            [float(plan.axis_knots[axis][index[axis]]) for axis in range(3)]
            for index in vertex_indices
        ],
        dtype=np.float64,
    )
    for index, point in zip(vertex_indices, coordinates, strict=True):
        for axis in range(3):
            if (
                Fraction.from_float(float(point[axis]))
                != plan.axis_knots[axis][index[axis]]
            ):
                raise ValueError(
                    "Grid vertices must be exactly representable in binary64."
                )
    mesh = CellMesh(
        coordinates,
        (CellBlock("cells", "hexahedron", np.asarray(cells, dtype=np.int32)),),
        vertex_global_ids=np.arange(coordinates.shape[0], dtype=np.int64),
        numeric_version=source_mesh.numeric_version,
    )
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    if np.any(np.asarray(validity.status) != int(CellValidityStatus.CERTIFIED_VALID)):
        raise ValueError(
            "Orthogonal PLC grid contains a cell without certified positive validity."
        )
    global_embedding = _require_global_embedding(
        mesh, geometry, validity, "Orthogonal hexahedral output"
    )
    cell_region_rows = np.asarray(output_regions, dtype=np.int64)
    integer_cells = np.asarray([index for index, _ in occupied], dtype=np.int64)
    construction_id = canonical_fingerprint(
        {
            "kind": "orthogonal-plc-hex-construction",
            "complex": source.complex_id,
            "source_topology": source_mesh.topology_id,
            "plan": plan.plan_id,
            "mesh": mesh.mesh_id,
            "regions": cell_region_rows,
            "integer_cells": integer_cells,
            "global_embedding": global_embedding.certificate_id,
        }
    )
    return OrthogonalPlcHexConstruction(
        mesh,
        geometry,
        validity,
        global_embedding,
        cell_region_rows,
        integer_cells,
        np.asarray(quality_scaled),
        np.asarray(quality_mean),
        np.asarray(quality_aspect),
        source_mesh.mesh_id,
        source.complex_id,
        plan.plan_id,
        construction_id,
    )


@dataclass(frozen=True, slots=True)
class _Plane:
    key: tuple[object, ...]
    normal: Point3
    rhs: Fraction


@dataclass(frozen=True, slots=True)
class _CutVertex:
    coordinate: Point3
    key: tuple[object, ...]
    support: tuple[int, ...]
    active_planes: tuple[tuple[object, ...], ...]
    weights: Weights


@dataclass(frozen=True, slots=True)
class _ClippedPolyhedron:
    source_cell_row: int
    grid_cell: tuple[int, int, int]
    vertices: tuple[_CutVertex, ...]
    faces: tuple[tuple[tuple[object, ...], tuple[int, ...]], ...]
    volume_six: Fraction


def _determinant3(
    first: Point3,
    second: Point3,
    third: Point3,
    /,
) -> Fraction:
    return (
        first[0] * (second[1] * third[2] - second[2] * third[1])
        - first[1] * (second[0] * third[2] - second[2] * third[0])
        + first[2] * (second[0] * third[1] - second[1] * third[0])
    )


def _dot3(first: Point3, second: Point3, /) -> Fraction:
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def _cross3(first: Point3, second: Point3, /) -> Point3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _subtract3(first: Point3, second: Point3, /) -> Point3:
    return (
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    )


def _mean_point(points: Sequence[Point3], /) -> Point3:
    count = len(points)
    if count == 0:
        raise ValueError("Exact point averaging requires at least one point.")
    return (
        sum((point[0] for point in points), Fraction(0)) / count,
        sum((point[1] for point in points), Fraction(0)) / count,
        sum((point[2] for point in points), Fraction(0)) / count,
    )


def _frame_point(
    point: Point3,
    frame_axes: tuple[tuple[Fraction, Fraction, Fraction], ...],
    frame_origin: Point3,
    /,
) -> Point3:
    offset = _subtract3(point, frame_origin)
    return (
        _dot3(frame_axes[0], offset),
        _dot3(frame_axes[1], offset),
        _dot3(frame_axes[2], offset),
    )


def _edge_key(first: int, second: int, /) -> tuple[int, int]:
    return (first, second) if first <= second else (second, first)


def _face_key3(vertices: tuple[int, int, int], /) -> tuple[int, int, int]:
    ordered = sorted(vertices)
    return (ordered[0], ordered[1], ordered[2])


def _ordered_cut_node_pair(
    first: int,
    second: int,
    node_keys: Sequence[tuple[object, ...]],
    /,
) -> tuple[int, int]:
    return (
        (first, second)
        if repr(node_keys[first]) <= repr(node_keys[second])
        else (second, first)
    )


def _solve_plane_triple(
    first: _Plane,
    second: _Plane,
    third: _Plane,
    /,
) -> Point3 | None:
    rows: tuple[Point3, Point3, Point3] = (
        first.normal,
        second.normal,
        third.normal,
    )
    rhs = ((first.rhs,), (second.rhs,), (third.rhs,))
    try:
        solution = solve_exact(rows, rhs)
    except ValueError:
        return None
    return (solution[0][0], solution[1][0], solution[2][0])


def _inside_planes(
    point: Point3,
    planes: tuple[_Plane, ...],
    /,
) -> bool:
    return all(_dot3(plane.normal, point) <= plane.rhs for plane in planes)


def _source_barycentric_support(
    point: Point3,
    corners: tuple[Point3, Point3, Point3, Point3],
    /,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    origin, first, second, third = corners
    e1 = _subtract3(first, origin)
    e2 = _subtract3(second, origin)
    e3 = _subtract3(third, origin)
    offset = _subtract3(point, origin)
    determinant = _determinant3(e1, e2, e3)
    if determinant == 0:
        raise ValueError("Cut-cell source tetrahedron is exactly degenerate.")
    w1 = _determinant3(offset, e2, e3) / determinant
    w2 = _determinant3(e1, offset, e3) / determinant
    w3 = _determinant3(e1, e2, offset) / determinant
    return (Fraction(1) - w1 - w2 - w3, w1, w2, w3)


def _tetrahedron_planes(
    source_cells: np.ndarray,
    cell_row: int,
    faces: np.ndarray,
    face_ids: np.ndarray,
    face_lookup: dict[tuple[int, int, int], int],
    exact_points: tuple[Point3, ...],
    /,
) -> tuple[_Plane, ...]:
    cell = source_cells[cell_row]
    local_faces = (
        ((1, 2, 3), 0),
        ((0, 3, 2), 1),
        ((0, 1, 3), 2),
        ((0, 2, 1), 3),
    )
    planes = []
    for local, opposite in local_faces:
        vertices: tuple[int, int, int] = (
            int(cell[local[0]]),
            int(cell[local[1]]),
            int(cell[local[2]]),
        )
        first = exact_points[vertices[0]]
        second = exact_points[vertices[1]]
        third = exact_points[vertices[2]]
        normal = _cross3(_subtract3(second, first), _subtract3(third, first))
        rhs = _dot3(normal, first)
        interior = exact_points[int(cell[opposite])]
        if _dot3(normal, interior) > rhs:
            normal = (-normal[0], -normal[1], -normal[2])
            rhs = -rhs
        face_row = face_lookup[_face_key3(vertices)]
        planes.append(_Plane(("source-face", int(face_ids[face_row])), normal, rhs))
    return tuple(planes)


def _canonical_plane_key(
    normal: tuple[Fraction, Fraction, Fraction],
    rhs: Fraction,
    /,
) -> tuple[object, ...]:
    pivot = next((value for value in normal if value), None)
    if pivot is None:
        raise ValueError("An exact grid plane must have a nonzero normal.")
    return (
        "grid",
        tuple(value / pivot for value in normal),
        rhs / pivot,
    )


def _grid_box_planes(
    grid_cell: tuple[int, int, int],
    plan: BalancedCutGridPlan,
    /,
    *,
    axis_knots: tuple[tuple[Fraction, ...], ...] | None = None,
    frame_axes: tuple[tuple[Fraction, Fraction, Fraction], ...] | None = None,
    frame_origin: tuple[Fraction, Fraction, Fraction] | None = None,
) -> tuple[_Plane, ...]:
    knots = plan.axis_knots if axis_knots is None else axis_knots
    axes = plan.frame_axes if frame_axes is None else frame_axes
    origin = plan.frame_origin if frame_origin is None else frame_origin
    planes = []
    for axis, cell in enumerate(grid_cell):
        lower = knots[axis][cell]
        upper = knots[axis][cell + 1]
        normal = axes[axis]
        offset = _dot3(normal, origin)
        negative: Point3 = (-normal[0], -normal[1], -normal[2])
        lower_rhs = -lower - offset
        upper_rhs = upper + offset
        planes.append(
            _Plane(_canonical_plane_key(negative, lower_rhs), negative, lower_rhs)
        )
        planes.append(_Plane(_canonical_plane_key(normal, upper_rhs), normal, upper_rhs))
    return tuple(planes)


def _cut_vertex_identity(
    point: tuple[Fraction, Fraction, Fraction],
    support: tuple[int, ...],
    active: tuple[_Plane, ...],
    source_mesh: CellMesh,
    cell_row: int,
    source_cell_ids: np.ndarray,
    edge_lookup: dict[tuple[int, int], int],
    face_lookup: dict[tuple[int, int, int], int],
    edge_ids: np.ndarray,
    face_ids: np.ndarray,
    /,
) -> tuple[object, ...]:
    vertex_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
    if len(support) == 1:
        return ("cut-vertex", "vertex", int(vertex_ids[support[0]]), ())
    if len(support) == 2:
        entity = edge_lookup[_edge_key(support[0], support[1])]
        stratum = ("edge", int(edge_ids[entity]))
    elif len(support) == 3:
        entity = face_lookup[_face_key3((support[0], support[1], support[2]))]
        stratum = ("face", int(face_ids[entity]))
    else:
        stratum = ("cell", int(source_cell_ids[cell_row]))
    grid_planes = tuple(
        sorted(
            (plane.key for plane in active if plane.key[0] == "grid"),
            key=repr,
        )
    )
    return ("cut-vertex", stratum[0], stratum[1], grid_planes)


def _clip_tetrahedron_to_grid_cell(
    source_mesh: CellMesh,
    source_cells: np.ndarray,
    source_cell_ids: np.ndarray,
    source_cell_row: int,
    grid_cell: tuple[int, int, int],
    plan: BalancedCutGridPlan,
    exact_points: tuple[Point3, ...],
    source_cell_planes: tuple[_Plane, ...],
    edge_lookup: dict[tuple[int, int], int],
    face_lookup: dict[tuple[int, int, int], int],
    edge_ids: np.ndarray,
    face_ids: np.ndarray,
    /,
    *,
    axis_knots: tuple[tuple[Fraction, ...], ...] | None = None,
    frame_axes: tuple[tuple[Fraction, Fraction, Fraction], ...] | None = None,
    frame_origin: tuple[Fraction, Fraction, Fraction] | None = None,
) -> _ClippedPolyhedron | None:
    planes = (
        *source_cell_planes,
        *_grid_box_planes(
            grid_cell,
            plan,
            axis_knots=axis_knots,
            frame_axes=frame_axes,
            frame_origin=frame_origin,
        ),
    )
    intersections: set[tuple[Fraction, Fraction, Fraction]] = set()
    for triple in combinations(planes, 3):
        point = _solve_plane_triple(*triple)
        if point is not None and _inside_planes(point, planes):
            intersections.add(point)
    if len(intersections) < 4:
        return None
    source_cell = source_cells[source_cell_row]
    corners: tuple[Point3, Point3, Point3, Point3] = (
        exact_points[int(source_cell[0])],
        exact_points[int(source_cell[1])],
        exact_points[int(source_cell[2])],
        exact_points[int(source_cell[3])],
    )
    vertices: list[_CutVertex] = []
    for point in sorted(intersections):
        barycentric = _source_barycentric_support(point, corners)
        weights: Weights = tuple(
            sorted(
                (int(source_cell[index]), weight)
                for index, weight in enumerate(barycentric)
                if weight
            )
        )
        support: tuple[int, ...] = tuple(vertex for vertex, _ in weights)
        active = tuple(
            plane for plane in planes if _dot3(plane.normal, point) == plane.rhs
        )
        key = _cut_vertex_identity(
            point,
            support,
            active,
            source_mesh,
            source_cell_row,
            source_cell_ids,
            edge_lookup,
            face_lookup,
            edge_ids,
            face_ids,
        )
        vertices.append(
            _CutVertex(
                point,
                key,
                support,
                tuple(plane.key for plane in active),
                weights,
            )
        )
    vertex_coordinates = tuple(vertex.coordinate for vertex in vertices)
    face_by_vertices: dict[tuple[object, ...], tuple[int, ...]] = {}
    for plane in planes:
        active_rows = tuple(
            row
            for row, vertex in enumerate(vertices)
            if plane.key in vertex.active_planes
        )
        if len(active_rows) < 3:
            continue
        loop = projected_convex_hull(
            active_rows, vertex_coordinates, plane.normal, plane.rhs
        )
        face_key = tuple(sorted((vertices[row].key for row in loop), key=repr))
        previous = face_by_vertices.get(face_key)
        if previous is not None and set(previous) != set(loop):
            raise ValueError(
                "Coincident clipped PLC plane keys disagree on their face arrangement."
            )
        face_by_vertices[face_key] = loop
    faces = tuple(
        (face_key, loop)
        for face_key, loop in sorted(
            face_by_vertices.items(), key=lambda entry: repr(entry[0])
        )
    )
    if len(faces) < 4:
        return None
    center = _mean_point(tuple(vertex.coordinate for vertex in vertices))
    volume_six = Fraction(0)
    for _, loop in faces:
        first = vertices[loop[0]].coordinate
        for index in range(1, len(loop) - 1):
            second = vertices[loop[index]].coordinate
            third = vertices[loop[index + 1]].coordinate
            volume_six += abs(
                _determinant3(
                    _subtract3(first, center),
                    _subtract3(second, center),
                    _subtract3(third, center),
                )
            )
    if volume_six == 0:
        return None
    return _ClippedPolyhedron(
        source_cell_row,
        grid_cell,
        tuple(vertices),
        faces,
        volume_six,
    )


@final
class BalancedCutHexConstruction(StrictModule, NonTrainableState):
    """Exact cut-cell all-hex construction and independent cell-family evidence."""

    mesh: CellMesh
    geometry: CellGeometrySpec
    validity: CellValidityCertificate
    global_embedding: GlobalEmbeddingCertificate
    cell_regions: np.ndarray
    parent_offsets: np.ndarray
    parent_cells: np.ndarray
    parent_cell_ids: np.ndarray
    vertex_supports: np.ndarray
    cut_vertex_keys: tuple[tuple[object, ...], ...]
    integer_cells: np.ndarray
    integer_blocks: np.ndarray
    scaled_jacobian_sample_min: np.ndarray
    mean_ratio_sample_min: np.ndarray
    aspect_ratio_sample_max: np.ndarray
    source_mesh_id: str = eqx.field(static=True)
    source_complex_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    source_validity_id: str = eqx.field(static=True)
    reference_chart_determinant_bounds: tuple[tuple[Fraction, Fraction], ...] = eqx.field(
        static=True
    )
    source_reference_coverage: tuple[Fraction, ...] = eqx.field(static=True)
    source_region_ids: tuple[str, ...] = eqx.field(static=True)
    source_region_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    source_plc_vertex_ids: tuple[str, ...] = eqx.field(static=True)
    source_plc_vertex_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(
        static=True
    )
    source_plc_edge_ids: tuple[str, ...] = eqx.field(static=True)
    source_plc_edge_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    source_plc_facet_ids: tuple[str, ...] = eqx.field(static=True)
    source_plc_facet_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(
        static=True
    )
    construction_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        validity: CellValidityCertificate,
        global_embedding: GlobalEmbeddingCertificate,
        cell_regions: ConvertibleToArray,
        parent_offsets: ConvertibleToArray,
        parent_cells: ConvertibleToArray,
        parent_cell_ids: ConvertibleToArray,
        vertex_supports: ConvertibleToArray,
        cut_vertex_keys: Sequence[tuple[object, ...]],
        source_region_ids: tuple[str, ...],
        source_region_occurrence_paths: tuple[tuple[str, ...], ...],
        source_plc_vertex_ids: tuple[str, ...],
        source_plc_vertex_occurrence_paths: tuple[tuple[str, ...], ...],
        source_plc_edge_ids: tuple[str, ...],
        source_plc_edge_occurrence_paths: tuple[tuple[str, ...], ...],
        source_plc_facet_ids: tuple[str, ...],
        source_plc_facet_occurrence_paths: tuple[tuple[str, ...], ...],
        integer_cells: ConvertibleToArray,
        integer_blocks: ConvertibleToArray,
        scaled_jacobian_sample_min: ConvertibleToArray,
        mean_ratio_sample_min: ConvertibleToArray,
        aspect_ratio_sample_max: ConvertibleToArray,
        source_mesh_id: str,
        source_complex_id: str,
        plan_id: str,
        source_geometry_id: str,
        source_validity_id: str,
        reference_chart_determinant_bounds: Sequence[tuple[Fraction, Fraction]],
        source_reference_coverage: Sequence[Fraction],
        construction_id: str,
        /,
    ) -> None:
        regions = np.asarray(cell_regions, dtype=np.int64)
        offsets = np.asarray(parent_offsets, dtype=np.int64)
        parents = np.asarray(parent_cells, dtype=np.int64)
        parent_ids = np.asarray(parent_cell_ids, dtype=np.int64)
        vertex_keys = tuple(tuple(key) for key in cut_vertex_keys)
        region_ids = tuple(source_region_ids)
        supports = np.asarray(vertex_supports, dtype=np.int64)
        integer = np.asarray(integer_cells, dtype=np.int64)
        blocks = np.asarray(integer_blocks, dtype=np.int64)
        scaled = np.asarray(scaled_jacobian_sample_min, dtype=np.float64)
        mean = np.asarray(mean_ratio_sample_min, dtype=np.float64)
        aspect = np.asarray(aspect_ratio_sample_max, dtype=np.float64)
        chart_bounds = tuple(
            (Fraction(lower), Fraction(upper))
            for lower, upper in reference_chart_determinant_bounds
        )
        reference_coverage = tuple(Fraction(value) for value in source_reference_coverage)
        if not isinstance(global_embedding, GlobalEmbeddingCertificate):
            raise TypeError("global_embedding must be a GlobalEmbeddingCertificate.")
        if (
            global_embedding.status != "certified"
            or global_embedding.validity.certificate_id != validity.certificate_id
        ):
            raise ValueError(
                "Balanced cut global embedding must bind its current validity."
            )
        global_embedding.binding.require(mesh, geometry)
        connectivity = mesh.connectivity
        if not isinstance(connectivity, HexahedralConnectivity):
            raise ValueError(
                "Balanced cut construction requires hexahedral connectivity."
            )
        count = connectivity.cell_count
        if mesh.topological_dimension != 3 or any(
            block.cell_kind != "hexahedron" for block in mesh.blocks
        ):
            raise ValueError("Balanced cut construction must contain only hexahedra.")
        if (
            regions.shape != (count,)
            or np.any((regions < 0) | (regions >= len(region_ids)))
            or offsets.shape != (count + 1,)
            or offsets[0] != 0
            or offsets[-1] != parents.size
            or np.any(offsets[1:] <= offsets[:-1])
            or parents.shape != (count,)
            or parent_ids.shape != (count,)
            or np.any(parent_ids < 0)
            or integer.shape != (count, 3)
            or blocks.shape != (count,)
            or np.any(blocks < 0)
            or supports.shape != (mesh.coordinates.shape[0], 4)
            or len(vertex_keys) != mesh.coordinates.shape[0]
            or any(value.shape != (count,) for value in (scaled, mean, aspect))
            or len(chart_bounds) != count
            or any(lower <= 0 or upper < lower for lower, upper in chart_bounds)
            or not reference_coverage
            or any(value != Fraction(1, 6) for value in reference_coverage)
            or not region_ids
            or any(not value for value in region_ids)
            or len(set(region_ids)) != len(region_ids)
            or len(source_region_occurrence_paths) != len(region_ids)
            or len(source_plc_vertex_occurrence_paths) != len(source_plc_vertex_ids)
            or len(source_plc_edge_occurrence_paths) != len(source_plc_edge_ids)
            or len(source_plc_facet_occurrence_paths) != len(source_plc_facet_ids)
            or any(not value for value in source_plc_vertex_ids)
            or any(not value for value in source_plc_edge_ids)
            or any(not value for value in source_plc_facet_ids)
        ):
            raise ValueError("Cut-cell ancestry, region, and quality banks must align.")
        if np.any(supports < -1) or np.any(np.all(supports < 0, axis=1)):
            raise ValueError(
                "Cut-cell vertex supports must name source simplex vertices."
            )
        if not all(np.all(np.isfinite(value)) for value in (scaled, mean, aspect)):
            raise ValueError("Cut-cell sampled quality evidence must be finite.")
        if not all(
            (
                source_mesh_id,
                source_complex_id,
                plan_id,
                source_geometry_id,
                source_validity_id,
                construction_id,
            )
        ):
            raise ValueError("Balanced cut construction identities must be non-empty.")
        self.mesh = mesh
        self.geometry = geometry
        self.validity = validity
        self.global_embedding = global_embedding
        self.cell_regions = regions
        self.parent_offsets = offsets
        self.parent_cells = parents
        self.parent_cell_ids = parent_ids
        self.vertex_supports = supports
        self.cut_vertex_keys = vertex_keys
        self.integer_cells = integer
        self.integer_blocks = blocks
        self.scaled_jacobian_sample_min = scaled
        self.mean_ratio_sample_min = mean
        self.aspect_ratio_sample_max = aspect
        self.source_mesh_id = source_mesh_id
        self.source_complex_id = source_complex_id
        self.plan_id = plan_id
        self.source_geometry_id = source_geometry_id
        self.source_validity_id = source_validity_id
        self.reference_chart_determinant_bounds = chart_bounds
        self.source_reference_coverage = reference_coverage
        self.source_region_ids = region_ids
        self.source_region_occurrence_paths = source_region_occurrence_paths
        self.source_plc_vertex_ids = source_plc_vertex_ids
        self.source_plc_vertex_occurrence_paths = source_plc_vertex_occurrence_paths
        self.source_plc_edge_ids = source_plc_edge_ids
        self.source_plc_edge_occurrence_paths = source_plc_edge_occurrence_paths
        self.source_plc_facet_ids = source_plc_facet_ids
        self.source_plc_facet_occurrence_paths = source_plc_facet_occurrence_paths
        self.construction_id = construction_id


@final
class _CutCoordinateMap(StrictModule, NonTrainableState):
    """Bind exact output-vertex images to the retained parent coefficient bank."""

    exact_vertices: tuple[ExactPoint3, ...] = eqx.field(static=True)
    rounded_vertices: np.ndarray
    source_complex: PiecewiseLinearComplex
    plan: BalancedCutGridPlan
    schedule: NativeHexGridSchedule
    source_mesh: CellMesh
    cell_regions: np.ndarray
    source_geometry: CellGeometrySpec
    target_mesh_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    node_keys: tuple[tuple[object, ...], ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_map_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_id: str,
        source_revision: str,
        source: PiecewiseLinearComplex,
        plan: BalancedCutGridPlan,
        schedule: NativeHexGridSchedule,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        cell_regions: ConvertibleToArray,
        target_mesh: CellMesh,
        node_keys: tuple[tuple[object, ...], ...],
        exact_vertices: tuple[ExactPoint3, ...],
        /,
    ) -> None:
        if (
            not isinstance(source, PiecewiseLinearComplex)
            or not isinstance(plan, BalancedCutGridPlan)
            or not isinstance(schedule, NativeHexGridSchedule)
        ):
            raise TypeError(
                "Cut-coordinate replay requires its PLC, grid plan, and schedule."
            )
        if not isinstance(source_mesh, CellMesh) or not isinstance(target_mesh, CellMesh):
            raise TypeError("Cut-coordinate replay requires source and target meshes.")
        if not isinstance(source_geometry, CellGeometrySpec):
            raise TypeError("source_geometry must be CellGeometrySpec.")
        source_geometry.resolve(source_mesh)
        source_map = source_geometry.source_map
        if source_map is not None and (
            source_map.source_id != source_id
            or source_map.source_revision != source_revision
        ):
            raise ValueError(
                "Cut-coordinate replay source map changed its owner or revision."
            )
        raw_regions = np.asarray(cell_regions)
        region_count = sum(block.cell_count for block in source_mesh.blocks)
        if (
            raw_regions.dtype.kind not in "iu"
            or raw_regions.shape != (region_count,)
            or np.any(raw_regions < 0)
            or np.any(raw_regions >= len(source.region_ids))
        ):
            raise ValueError(
                "Cut-coordinate replay requires every actual source cell region assignment."
            )
        regions = np.array(raw_regions, dtype=np.int64, copy=True)
        regions.setflags(write=False)
        if (
            source.complex_id != plan.source_complex_id
            or source_mesh.mesh_id != plan.source_mesh_id
            or source_mesh.topology_id != plan.source_topology_id
            or source_id != plan.source_id
            or source_revision != plan.source_revision
            or schedule.schedule_id != plan.schedule_id
        ):
            raise ValueError(
                "Cut-coordinate replay inputs disagree with the balanced-grid plan."
            )
        points = np.array(target_mesh.coordinates, dtype=np.float64, copy=True)
        if (
            len(node_keys) != points.shape[0]
            or len(exact_vertices) != len(node_keys)
            or any(
                len(point) != 3
                or any(
                    not isinstance(value, (Fraction, RadicalElement)) for value in point
                )
                for point in exact_vertices
            )
            or not np.array_equal(
                points,
                np.asarray(
                    [[float(value) for value in point] for point in exact_vertices],
                    dtype=np.float64,
                ),
            )
        ):
            raise ValueError("Exact cut coordinates must match every target mesh vertex.")
        points.setflags(write=False)
        self.rounded_vertices = points
        self.exact_vertices = exact_vertices
        self.source_complex = source
        self.plan = plan
        self.schedule = schedule
        self.source_mesh = source_mesh
        self.source_geometry = source_geometry
        self.cell_regions = regions
        self.target_mesh_id = target_mesh.mesh_id
        self.target_topology_id = target_mesh.topology_id
        self.node_keys = node_keys
        self.source_id = source_id
        self.source_revision = source_revision
        self.source_map_id = canonical_fingerprint(
            {
                "kind": "exact-balanced-cut-coordinate-map",
                "source_id": source_id,
                "source_revision": source_revision,
                "source_complex": source.complex_id,
                "source_mesh": source_mesh.mesh_id,
                "source_topology": source_mesh.topology_id,
                "schedule": schedule.schedule_id,
                "source_geometry": cell_geometry_id(source_geometry),
                "source_coordinate_map": None
                if source_map is None
                else source_map.source_map_id,
                "plan": plan.plan_id,
                "cell_regions": regions,
                "target_mesh": target_mesh.mesh_id,
                "target_topology": target_mesh.topology_id,
                "node_keys": tuple(repr(key) for key in node_keys),
                "exact_vertices": tuple(
                    tuple(_exact_scalar_key(value) for value in point)
                    for point in exact_vertices
                ),
            }
        )

    @classmethod
    def from_stored_fields(
        cls, value: _CutCoordinateMap, target_mesh: CellMesh, /
    ) -> _CutCoordinateMap:
        if type(value) is not cls:
            raise TypeError("Stored cut coordinates must be _CutCoordinateMap.")
        if (
            target_mesh.mesh_id != value.target_mesh_id
            or target_mesh.topology_id != value.target_topology_id
        ):
            raise ValueError("Stored cut coordinates bind a different target mesh.")
        renewed_plan = BalancedCutGridPlan(
            value.plan.axis_knots,
            value.source_complex,
            value.source_mesh,
            value.schedule,
            value.plan.target_size,
            source_id=value.source_id,
            source_revision=value.source_revision,
            frame_axes=np.asarray(value.plan.frame_axes, dtype=np.float64),
            frame_origin=np.asarray(value.plan.frame_origin, dtype=np.float64),
            frame_id=value.plan.frame_id,
            frame_layout=value.plan.frame_layout,
        )
        if (
            renewed_plan.plan_id != value.plan.plan_id
            or renewed_plan.schedule_id != value.plan.schedule_id
        ):
            raise ValueError(
                "Stored balanced-grid plan does not replay from its exact inputs."
            )
        replayed = generate_balanced_cut_hexes(
            value.source_complex,
            value.source_mesh,
            value.cell_regions,
            renewed_plan,
            value.schedule,
            source_geometry=value.source_geometry,
        )
        replayed_map = replayed.geometry.source_map
        if (
            not isinstance(replayed_map, cls)
            or replayed_map.source_map_id != value.source_map_id
            or replayed_map.node_keys != value.node_keys
            or replayed_map.exact_vertices != value.exact_vertices
            or not np.array_equal(replayed_map.cell_regions, value.cell_regions)
            or replayed.mesh.mesh_id != target_mesh.mesh_id
            or replayed.mesh.topology_id != target_mesh.topology_id
            or not np.array_equal(
                np.asarray(replayed.mesh.coordinates),
                np.asarray(target_mesh.coordinates),
            )
        ):
            raise ValueError(
                "Stored cut coordinates do not reproduce from their source and grid plan."
            )
        renewed = cls(
            value.source_id,
            value.source_revision,
            value.source_complex,
            renewed_plan,
            value.schedule,
            value.source_mesh,
            value.source_geometry,
            value.cell_regions,
            target_mesh,
            value.node_keys,
            value.exact_vertices,
        )
        if (
            renewed.source_map_id != value.source_map_id
            or not np.array_equal(renewed.rounded_vertices, value.rounded_vertices)
            or not np.array_equal(renewed.cell_regions, value.cell_regions)
        ):
            raise ValueError(
                "Stored cut-coordinate replay fields do not renew their identity."
            )
        return renewed

    def source_coordinates(
        self, indices: ConvertibleToArray | None = None, /
    ) -> tuple[ExactPoint3, ...]:
        """Expose exact parent controls; output vertex images remain in exact_vertices."""
        selected = None if indices is None else np.asarray(indices)
        return cast(
            tuple[ExactPoint3, ...],
            self.source_geometry.source_coordinates(selected),
        )


class _CutNodeBank:
    def __init__(self, source_mesh: CellMesh, /) -> None:
        source_points = np.asarray(source_mesh.coordinates, dtype=np.float64)
        self.coordinates: list[Point3] = [
            _fraction_point3(point) for point in source_points
        ]
        self.weights: list[Weights] = [
            ((row, Fraction(1)),) for row in range(source_points.shape[0])
        ]
        self.supports: list[tuple[int, ...]] = [
            (row,) for row in range(source_points.shape[0])
        ]
        vertex_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
        self.rows: dict[tuple[object, ...], int] = {
            ("cut-vertex", "vertex", int(identifier), ()): row
            for row, identifier in enumerate(vertex_ids.tolist())
        }
        self.keys: list[tuple[object, ...]] = [
            ("cut-vertex", "vertex", int(identifier), ())
            for identifier in vertex_ids.tolist()
        ]

    def add(
        self,
        key: tuple[object, ...],
        point: Point3,
        weights: Weights,
        /,
    ) -> int:
        normalized: Weights = tuple(
            (int(vertex), Fraction(weight)) for vertex, weight in weights if weight
        )
        support: tuple[int, ...] = tuple(vertex for vertex, _ in normalized)
        if not normalized or sum((weight for _, weight in normalized), Fraction(0)) != 1:
            raise ValueError(
                "Cut-node source reference weights must form an exact partition of unity."
            )
        row = self.rows.get(key)
        if row is not None:
            if (
                self.coordinates[row] != point
                or self.supports[row] != support
                or self.weights[row] != normalized
            ):
                raise ValueError(
                    "Exact cut arrangement ancestry disagrees on a shared vertex."
                )
            return row
        self.rows[key] = len(self.coordinates)
        self.keys.append(key)
        self.coordinates.append(point)
        self.weights.append(normalized)
        self.supports.append(support)
        return len(self.coordinates) - 1


def _average_source_weights(
    rows: tuple[int, ...],
    weights: list[Weights],
    /,
) -> Weights:
    totals: dict[int, Fraction] = defaultdict(Fraction)
    for row in rows:
        for vertex, value in weights[row]:
            totals[vertex] += value / len(rows)
    return tuple(sorted((vertex, value) for vertex, value in totals.items() if value))


def _cut_polyhedra_are_trivalent(
    polyhedra: Sequence[_ClippedPolyhedron],
    /,
) -> bool:
    for polyhedron in polyhedra:
        vertex_neighbors: list[set[int]] = [set() for _ in polyhedron.vertices]
        for _, loop in polyhedron.faces:
            if len(loop) < 3:
                raise ValueError(
                    "Exact cut-polyhedron dualization encountered a face with fewer than "
                    "three vertices."
                )
            for position, vertex in enumerate(loop):
                vertex_neighbors[vertex].update(
                    (loop[position - 1], loop[(position + 1) % len(loop)])
                )
        if any(len(neighbors) != 3 for neighbors in vertex_neighbors):
            return False
    return True


def _tetrahedralize_cut_polyhedra(
    node_bank: _CutNodeBank,
    polyhedra: Sequence[_ClippedPolyhedron],
    source_cell_ids: np.ndarray,
    /,
) -> list[_ClippedPolyhedron]:
    tetrahedra: list[tuple[tuple[int, int, int, int], int, tuple[int, int, int]]] = []
    for polyhedron in polyhedra:
        local_vertex_rows = tuple(
            node_bank.add(
                vertex.key,
                vertex.coordinate,
                vertex.weights,
            )
            for vertex in polyhedron.vertices
        )
        polyhedron_weights = _average_source_weights(local_vertex_rows, node_bank.weights)
        polyhedron_center = _mean_point(
            tuple(vertex.coordinate for vertex in polyhedron.vertices)
        )
        cell_center_row = node_bank.add(
            (
                "cut-polyhedron-center",
                int(source_cell_ids[polyhedron.source_cell_row]),
                polyhedron.grid_cell,
            ),
            polyhedron_center,
            polyhedron_weights,
        )
        covered_volume_six = Fraction(0)
        for face_key, loop in polyhedron.faces:
            face_vertex_rows = tuple(local_vertex_rows[row] for row in loop)
            face_weights = _average_source_weights(face_vertex_rows, node_bank.weights)
            face_center = _mean_point(
                tuple(polyhedron.vertices[row].coordinate for row in loop)
            )
            face_center_row = node_bank.add(
                ("cut-polyhedron-face-center", face_key),
                face_center,
                face_weights,
            )
            for first, second in zip(loop, (*loop[1:], loop[0]), strict=True):
                tetrahedron = (
                    cell_center_row,
                    face_center_row,
                    local_vertex_rows[first],
                    local_vertex_rows[second],
                )
                a, b, c, d = (node_bank.coordinates[row] for row in tetrahedron)
                volume_six = abs(
                    _determinant3(
                        _subtract3(b, a),
                        _subtract3(c, a),
                        _subtract3(d, a),
                    )
                )
                if volume_six == 0:
                    raise ValueError(
                        "Exact face-center coning produced a degenerate cut tetrahedron "
                        f"for source cell ID {int(source_cell_ids[polyhedron.source_cell_row])}."
                    )
                covered_volume_six += volume_six
                tetrahedra.append(
                    (
                        tetrahedron,
                        polyhedron.source_cell_row,
                        polyhedron.grid_cell,
                    )
                )
        if covered_volume_six != polyhedron.volume_six:
            raise ValueError(
                "Exact face-center coning failed clipped-polyhedron coverage for source "
                f"cell ID {int(source_cell_ids[polyhedron.source_cell_row])}."
            )

    tetrahedron_face_counts: dict[tuple[object, ...], int] = defaultdict(int)
    for tetrahedron, _, _ in tetrahedra:
        for omitted in range(4):
            face_rows = tuple(
                tetrahedron[index] for index in range(4) if index != omitted
            )
            face_key = tuple(sorted((node_bank.keys[row] for row in face_rows), key=repr))
            tetrahedron_face_counts[face_key] += 1
    if any(count > 2 for count in tetrahedron_face_counts.values()):
        raise ValueError(
            "Exact cut-polyhedron coning produced a nonmanifold tetrahedral face."
        )

    tetrahedra_as_polyhedra: list[_ClippedPolyhedron] = []
    for tetrahedron, source_cell_row, grid_cell in tetrahedra:
        vertices = tuple(
            _CutVertex(
                node_bank.coordinates[row],
                node_bank.keys[row],
                node_bank.supports[row],
                (),
                node_bank.weights[row],
            )
            for row in tetrahedron
        )
        faces = tuple(
            (
                tuple(
                    sorted(
                        (vertices[row].key for row in range(4) if row != omitted),
                        key=repr,
                    )
                ),
                tuple(row for row in range(4) if row != omitted),
            )
            for omitted in range(4)
        )
        a, b, c, d = (vertex.coordinate for vertex in vertices)
        volume_six = abs(
            _determinant3(
                _subtract3(b, a),
                _subtract3(c, a),
                _subtract3(d, a),
            )
        )
        tetrahedra_as_polyhedra.append(
            _ClippedPolyhedron(
                source_cell_row,
                grid_cell,
                vertices,
                faces,
                volume_six,
            )
        )
    return tetrahedra_as_polyhedra


def _dualize_cut_polyhedra_to_hexes(
    node_bank: _CutNodeBank,
    polyhedra: Sequence[_ClippedPolyhedron],
    source_cell_ids: np.ndarray,
    region_rows: np.ndarray,
    plan: BalancedCutGridPlan,
    /,
) -> tuple[
    list[list[int]],
    list[int],
    list[int],
    list[tuple[int, int, int]],
    list[int],
]:
    polyhedron_face_counts: dict[tuple[object, ...], int] = defaultdict(int)
    edge_centers: dict[tuple[object, ...], int] = {}
    hexes: list[list[int]] = []
    output_regions: list[int] = []
    parents: list[int] = []
    integer_cells: list[tuple[int, int, int]] = []
    integer_blocks: list[int] = []
    reference_hex: tuple[tuple[int, int, int], ...] = tuple(
        (int(point[0]), int(point[1]), int(point[2]))
        for point in reference_cell_topology("hexahedron").vertices
    )

    def entity_key(rows: tuple[int, ...]) -> tuple[object, ...]:
        return tuple(sorted((node_bank.keys[row] for row in rows), key=repr))

    def weights_for_rows(rows: tuple[int, ...]) -> tuple[tuple[int, Fraction], ...]:
        return _average_source_weights(rows, node_bank.weights)

    for polyhedron in polyhedra:
        local_vertex_rows = tuple(
            node_bank.add(
                vertex.key,
                vertex.coordinate,
                vertex.weights,
            )
            for vertex in polyhedron.vertices
        )
        poly_weights = _average_source_weights(local_vertex_rows, node_bank.weights)
        poly_center = _mean_point(
            tuple(vertex.coordinate for vertex in polyhedron.vertices)
        )
        cell_center_row = node_bank.add(
            (
                "cut-polyhedron-center",
                int(source_cell_ids[polyhedron.source_cell_row]),
                polyhedron.grid_cell,
                entity_key(local_vertex_rows),
            ),
            poly_center,
            poly_weights,
        )
        vertex_neighbors: list[set[int]] = [set() for _ in local_vertex_rows]
        vertex_face_centers: list[dict[tuple[int, int], int]] = [
            {} for _ in local_vertex_rows
        ]
        for face_key, loop in polyhedron.faces:
            if len(loop) < 3:
                raise ValueError(
                    "Exact cut-polyhedron dualization encountered a face with fewer "
                    "than three vertices."
                )
            polyhedron_face_counts[face_key] += 1
            face_vertex_rows = tuple(local_vertex_rows[row] for row in loop)
            face_weights = _average_source_weights(face_vertex_rows, node_bank.weights)
            face_center = _mean_point(
                tuple(polyhedron.vertices[row].coordinate for row in loop)
            )
            face_center_row = node_bank.add(
                ("cut-polyhedron-face-center", face_key),
                face_center,
                face_weights,
            )
            for face_position, local_vertex in enumerate(loop):
                vertex_row = local_vertex_rows[local_vertex]
                previous = local_vertex_rows[loop[face_position - 1]]
                following = local_vertex_rows[loop[(face_position + 1) % len(loop)]]
                vertex_neighbors[local_vertex].update((previous, following))
                pair = _ordered_cut_node_pair(previous, following, node_bank.keys)
                if pair in vertex_face_centers[local_vertex]:
                    raise ValueError(
                        "Exact cut-polyhedron dualization found repeated face "
                        "incidence at source cell ID "
                        f"{int(source_cell_ids[polyhedron.source_cell_row])}."
                    )
                vertex_face_centers[local_vertex][pair] = face_center_row
                for first, second in (
                    (vertex_row, previous),
                    (vertex_row, following),
                ):
                    key = entity_key((first, second))
                    if key in edge_centers:
                        continue
                    edge_center = _mean_point(
                        (
                            node_bank.coordinates[first],
                            node_bank.coordinates[second],
                        )
                    )
                    edge_centers[key] = node_bank.add(
                        ("cut-dual-edge-center", key),
                        edge_center,
                        weights_for_rows((first, second)),
                    )

        for local_vertex, vertex_row in enumerate(local_vertex_rows):
            incident_rows = tuple(
                sorted(
                    vertex_neighbors[local_vertex],
                    key=lambda row: repr(node_bank.keys[row]),
                )
            )
            if len(incident_rows) != 3:
                raise ValueError(
                    "Exact cut-polyhedron dualization requires trivalent clipped "
                    f"vertices; source cell ID "
                    f"{int(source_cell_ids[polyhedron.source_cell_row])} has valence "
                    f"{len(incident_rows)} at a clipped vertex."
                )
            expected_pairs = {
                _ordered_cut_node_pair(first, second, node_bank.keys)
                for first, second in combinations(incident_rows, 2)
            }
            face_center_by_pair = vertex_face_centers[local_vertex]
            if set(face_center_by_pair) != expected_pairs:
                raise ValueError(
                    "Exact cut-polyhedron dualization found an incomplete vertex-face "
                    f"star at source cell ID {int(source_cell_ids[polyhedron.source_cell_row])}."
                )
            cube_rows_by_bits: dict[tuple[int, int, int], int] = {
                (0, 0, 0): vertex_row,
                (1, 0, 0): edge_centers[entity_key((vertex_row, incident_rows[0]))],
                (0, 1, 0): edge_centers[entity_key((vertex_row, incident_rows[1]))],
                (0, 0, 1): edge_centers[entity_key((vertex_row, incident_rows[2]))],
                (1, 1, 0): face_center_by_pair[
                    _ordered_cut_node_pair(
                        incident_rows[0], incident_rows[1], node_bank.keys
                    )
                ],
                (1, 0, 1): face_center_by_pair[
                    _ordered_cut_node_pair(
                        incident_rows[0], incident_rows[2], node_bank.keys
                    )
                ],
                (0, 1, 1): face_center_by_pair[
                    _ordered_cut_node_pair(
                        incident_rows[1], incident_rows[2], node_bank.keys
                    )
                ],
                (1, 1, 1): cell_center_row,
            }
            hexes.append([cube_rows_by_bits[bits] for bits in reference_hex])
            source_cell_row = polyhedron.source_cell_row
            parents.append(source_cell_row)
            output_regions.append(int(region_rows[source_cell_row]))
            integer_cells.append(polyhedron.grid_cell)
            integer_blocks.append(
                plan.frame_block_indices[source_cell_row]
                if plan.frame_layout_id is not None
                else 0
            )

    if any(count > 2 for count in polyhedron_face_counts.values()):
        raise ValueError(
            "Exact cut-polyhedron dualization produced a nonmanifold clipped face."
        )
    return hexes, output_regions, parents, integer_cells, integer_blocks


def _cut_trilinear_chart(
    vertices: tuple[Point3, ...],
    /,
) -> tuple[Expression, Expression, Expression]:
    variables = axes(3)
    one = constant(Fraction(1), 3)
    reference = reference_cell_topology("hexahedron").vertices
    result: list[Expression] = []
    for axis in range(3):
        component: Expression = constant(Fraction(0), 3)
        for row, bits in enumerate(reference):
            shape: Expression = constant(Fraction(1), 3)
            for direction in range(3):
                factor: Expression = (
                    variables[direction]
                    if int(bits[direction]) == 1
                    else add(one, scale(variables[direction], -1))
                )
                shape = multiply(shape, factor)
            component = add(component, scale(shape, vertices[row][axis]))
        result.append(component)
    return (result[0], result[1], result[2])


def _cut_chart_determinant(
    chart: tuple[Expression, Expression, Expression],
    /,
) -> Expression:
    jacobian = tuple(
        tuple(derivative(component, axis) for axis in range(3)) for component in chart
    )
    minor0 = add(
        multiply(jacobian[1][1], jacobian[2][2]),
        scale(multiply(jacobian[1][2], jacobian[2][1]), -1),
    )
    minor1 = add(
        multiply(jacobian[1][0], jacobian[2][2]),
        scale(multiply(jacobian[1][2], jacobian[2][0]), -1),
    )
    minor2 = add(
        multiply(jacobian[1][0], jacobian[2][1]),
        scale(multiply(jacobian[1][1], jacobian[2][0]), -1),
    )
    return add(
        add(
            multiply(jacobian[0][0], minor0),
            scale(multiply(jacobian[0][1], minor1), -1),
        ),
        multiply(jacobian[0][2], minor2),
    )


def _cut_chart_bounds(
    chart: tuple[Expression, Expression, Expression],
    cell_id: int,
    /,
) -> tuple[Fraction, Fraction]:
    determinant = _cut_chart_determinant(chart)
    if isinstance(determinant, RationalPolynomial):
        raise ValueError(
            f"Exact cut reference chart for output cell ID {cell_id} has a rational determinant."
        )
    raw_controls = bernstein_coefficients(determinant, "box", 3)
    controls: tuple[Fraction, ...] = tuple(
        control for control in raw_controls if isinstance(control, Fraction)
    )
    if len(controls) != len(raw_controls) or not controls or min(controls) <= 0:
        raise ValueError(
            f"Exact cut reference chart for output cell ID {cell_id} lacks "
            "a positive whole-cell Bernstein determinant bound."
        )
    return min(controls), max(controls)


def _cut_chart_volume(
    chart: tuple[Expression, Expression, Expression],
    /,
) -> Fraction:
    determinant = _cut_chart_determinant(chart)
    if isinstance(determinant, RationalPolynomial):
        raise ValueError(
            "Cut reference chart volume requires an exact polynomial determinant."
        )
    total = Fraction(0)
    for exponent, coefficient in determinant.items():
        rational_coefficient = _require_fraction(
            coefficient,
            "Cut reference chart volume requires rational polynomial coefficients.",
        )
        total += rational_coefficient * Fraction(
            1,
            (exponent[0] + 1) * (exponent[1] + 1) * (exponent[2] + 1),
        )
    return total


def _cut_source_reference_point(
    weights: Weights,
    source_cell_row: int,
    source_cells: np.ndarray,
    /,
) -> Point3:
    source_cell = tuple(int(value) for value in source_cells[source_cell_row])
    references = reference_cell_topology("tetrahedron").vertices
    local = {
        vertex: _fraction_point3(np.asarray(references[index], dtype=np.float64))
        for index, vertex in enumerate(source_cell)
    }
    if any(vertex not in local for vertex, _ in weights):
        raise ValueError("Cut-node barycentric weights leave their source tetrahedron.")
    return (
        sum((weight * local[vertex][0] for vertex, weight in weights), Fraction(0)),
        sum((weight * local[vertex][1] for vertex, weight in weights), Fraction(0)),
        sum((weight * local[vertex][2] for vertex, weight in weights), Fraction(0)),
    )


def _cut_quality_meets_schedule(quality: Any, schedule: NativeHexGridSchedule, /) -> bool:
    if not np.all(np.asarray(quality.sampled_valid, dtype=np.bool_)):
        return False
    scaled = np.asarray(quality.scaled_jacobian, dtype=np.float64)
    mean = np.asarray(quality.mean_ratios, dtype=np.float64)
    aspect = np.asarray(quality.aspect_ratios, dtype=np.float64)
    return bool(
        scaled.size
        and mean.size
        and aspect.size
        and float(np.min(scaled)) >= schedule.minimum_scaled_jacobian
        and float(np.min(mean)) >= schedule.minimum_mean_ratio
        and float(np.max(aspect)) <= schedule.maximum_aspect_ratio
    )


def _relocate_cut_nodes_in_root_coordinates(
    source_mesh: CellMesh,
    source_cells: np.ndarray,
    source_expressions: tuple[tuple[Expression, Expression, Expression], ...],
    node_bank: _CutNodeBank,
    hexes: list[list[int]],
    parents: list[int],
    schedule: NativeHexGridSchedule,
    /,
) -> tuple[bool, str]:
    """Propose metric relocation in exact affine root-image space.

    Nodes remain convex combinations of their authored source-vertex support;
    the caller still rebuilds the exact charts and certifies final geometry.
    """
    if (
        not len(source_cells)
        or len(source_expressions) != len(source_cells)
        or len(parents) != len(hexes)
        or any(len(cell) != 4 for cell in source_cells)
    ):
        return False, "source_cell_shape"

    def is_affine_exact(expression: Expression, /) -> bool:
        if isinstance(expression, RationalPolynomial):
            if len(expression.denominator) != 1:
                return False
            denominator_exponent, denominator_coefficient = next(
                iter(expression.denominator.items())
            )
            if (
                denominator_exponent != (0, 0, 0)
                or not isinstance(denominator_coefficient, Fraction)
                or denominator_coefficient == 0
            ):
                return False
            polynomial = expression.numerator
        else:
            polynomial = expression
        return all(
            isinstance(coefficient, Fraction) and sum(exponent) <= 1
            for exponent, coefficient in polynomial.items()
        )

    if any(
        not is_affine_exact(expression)
        for cell_expressions in source_expressions
        for expression in cell_expressions
    ):
        return False, "source_chart_not_affine"

    root_vertices: dict[int, Point3] = {}
    parent_row = 0
    for block in source_mesh.blocks:
        if block.cell_kind not in ("tetrahedron", "simplex:3"):
            return False, "unsupported_source_cell_kind"
        reference_vertices = reference_cell_topology(block.cell_kind).vertices
        for cell_row in range(block.cell_count):
            cell_expressions = source_expressions[parent_row]
            local_cell = tuple(int(value) for value in source_cells[parent_row])
            for local, source_row in enumerate(local_cell):
                reference_point = _fraction_point3(
                    np.asarray(reference_vertices[local], dtype=np.float64)
                )
                image = tuple(
                    evaluate_expression(expression, reference_point)
                    for expression in cell_expressions
                )
                if len(image) != 3 or not all(
                    isinstance(value, Fraction) for value in image
                ):
                    return False, "nonrational_source_vertex_image"
                exact_image = cast(Point3, image)
                previous = root_vertices.get(source_row)
                if previous is not None and previous != exact_image:
                    return False, "inconsistent_shared_source_vertex_image"
                root_vertices[source_row] = exact_image
            parent_row += 1
    if parent_row != len(source_cells) or not root_vertices:
        return False, "incomplete_source_root_vertex_bank"

    for node_row, weights in enumerate(node_bank.weights):
        if any(vertex not in root_vertices for vertex, _ in weights):
            return False, "node_support_outside_root_vertex_bank"
        exact_image = tuple(
            sum(
                (weight * root_vertices[vertex][axis] for vertex, weight in weights),
                Fraction(0),
            )
            for axis in range(3)
        )
        if exact_image != node_bank.coordinates[node_row]:
            return False, "node_exact_source_image_mismatch"
    cells = np.asarray(hexes, dtype=np.int64)
    root_coordinates = np.asarray(
        [
            [
                float(
                    sum(
                        (
                            weight * root_vertices[vertex][axis]
                            for vertex, weight in weights
                        ),
                        Fraction(0),
                    )
                )
                for axis in range(3)
            ]
            for weights in node_bank.weights
        ],
        dtype=np.float64,
    )
    cell_ids = np.arange(cells.shape[0], dtype=np.int64)
    vertex_ids = np.arange(root_coordinates.shape[0], dtype=np.int64)
    root_mesh = CellMesh(
        root_coordinates,
        (
            CellBlock(
                "cut-quality",
                "hexahedron",
                cells,
                global_ids=cell_ids,
            ),
        ),
        vertex_global_ids=vertex_ids,
        numeric_version=source_mesh.numeric_version,
    )
    initial_quality = evaluate_cell_quality(root_mesh)
    if not np.all(np.asarray(initial_quality.sampled_valid, dtype=np.bool_)):
        return False, "invalid_root_mesh_before_relocation"
    import jax
    import jax.numpy as jnp

    import phydrax.ein as ein

    from ..optim import Bounds, minimize, OptimizationTermination, ProjectedLBFGS

    fixed = np.asarray(
        [len(weights) == 1 for weights in node_bank.weights],
        dtype=np.bool_,
    )
    if bool(np.all(fixed)):
        return False, "no_movable_root_supports"
    support_points = np.zeros((root_coordinates.shape[0], 4, 3), dtype=np.float64)
    support_mask = np.zeros((root_coordinates.shape[0], 4), dtype=np.bool_)
    reference_vertices = tuple(
        tuple(Fraction(int(value)) for value in point)
        for point in reference_cell_topology("tetrahedron").vertices
    )
    chart_support_points = np.zeros((cells.shape[0], 8, 4, 3), dtype=np.float64)
    for output_row, (cell, parent) in enumerate(zip(cells, parents, strict=True)):
        parent_vertices = tuple(int(value) for value in source_cells[int(parent)])
        local_by_vertex = {vertex: local for local, vertex in enumerate(parent_vertices)}
        for local_cell_vertex, node_row in enumerate(cell.tolist()):
            weights = node_bank.weights[int(node_row)]
            if len(weights) > 4:
                return False, "invalid_parent_reference_support_arity"
            for local_support, (source_vertex, _) in enumerate(weights):
                source_local = local_by_vertex.get(source_vertex)
                if source_local is None:
                    return False, "node_support_outside_parent_tetrahedron"
                chart_support_points[output_row, local_cell_vertex, local_support] = (
                    tuple(float(value) for value in reference_vertices[source_local])
                )
    parameter_rows: list[int] = []
    parameter_columns: list[int] = []
    initial_parameters: list[float] = []
    for node_row, weights in enumerate(node_bank.weights):
        if not weights or len(weights) > 4:
            return False, "invalid_source_support_arity"
        reference_weight = weights[0][1]
        if reference_weight <= 0:
            return False, "invalid_source_support_weights"
        for local, (vertex, weight) in enumerate(weights):
            support_points[node_row, local] = tuple(
                float(value) for value in root_vertices[vertex]
            )
            support_mask[node_row, local] = True
            if local:
                ratio = float(weight / reference_weight)
                if not np.isfinite(ratio) or ratio <= 0.0:
                    return False, "invalid_source_support_logit"
                parameter_rows.append(node_row)
                parameter_columns.append(local)
                initial_parameters.append(float(np.log(ratio)))
    if not initial_parameters:
        return False, "no_variable_source_support_weights"

    support_points_jax = jnp.asarray(support_points, dtype=jnp.float64)
    support_mask_jax = jnp.asarray(support_mask, dtype=jnp.bool_)
    chart_support_points_jax = jnp.asarray(chart_support_points, dtype=jnp.float64)
    cell_vertices_jax = jnp.asarray(cells, dtype=jnp.int32)
    reference_bits = tuple(
        tuple(int(value) for value in point)
        for point in reference_cell_topology("hexahedron").vertices
    )
    determinant_grid = tuple(
        tuple(float(value) for value in point)
        for point in product((0.0, 0.5, 1.0), repeat=3)
    )
    shape_gradients = np.empty((len(determinant_grid), 8, 3), dtype=np.float64)
    for sample_row, point in enumerate(determinant_grid):
        for vertex_row, bits in enumerate(reference_bits):
            for axis in range(3):
                derivative_value = 1.0 if bits[axis] else -1.0
                for other_axis in range(3):
                    if other_axis == axis:
                        continue
                    derivative_value *= (
                        point[other_axis] if bits[other_axis] else 1.0 - point[other_axis]
                    )
                shape_gradients[sample_row, vertex_row, axis] = derivative_value
    shape_gradients_jax = jnp.asarray(shape_gradients, dtype=jnp.float64)
    bernstein_inverse_jax = jnp.asarray(
        ((1.0, 0.0, 0.0), (-0.5, 2.0, -0.5), (0.0, 0.0, 1.0)),
        dtype=jnp.float64,
    )
    chart_positive_margin = 64.0 * np.finfo(np.float64).eps
    base_logits = jnp.zeros((root_coordinates.shape[0], 4), dtype=jnp.float64)
    parameter_rows_jax = jnp.asarray(parameter_rows, dtype=jnp.int32)
    parameter_columns_jax = jnp.asarray(parameter_columns, dtype=jnp.int32)

    def probabilities_from_logits(parameters: Any, /) -> Any:
        logits = base_logits.at[parameter_rows_jax, parameter_columns_jax].set(parameters)
        logits = jnp.where(support_mask_jax, logits, -jnp.inf)
        probabilities = jax.nn.softmax(logits, axis=1)
        return probabilities / jnp.sum(probabilities, axis=1, keepdims=True)

    initial_parameters_jax = jnp.asarray(initial_parameters, dtype=jnp.float64)

    def quality_deficit(parameters: Any, /) -> Any:
        probabilities = probabilities_from_logits(parameters)
        coordinates = jnp.sum(probabilities[:, :, None] * support_points_jax, axis=1)
        quality = evaluate_cell_quality(root_mesh, coordinates)
        scaled_deficit = jnp.maximum(
            0.0,
            schedule.minimum_scaled_jacobian / quality.scaled_jacobian - 1.0,
        )
        mean_deficit = jnp.maximum(
            0.0,
            schedule.minimum_mean_ratio / quality.mean_ratios - 1.0,
        )
        aspect_deficit = jnp.maximum(
            0.0,
            quality.aspect_ratios / schedule.maximum_aspect_ratio - 1.0,
        )
        source_reference_points = jnp.sum(
            probabilities[cell_vertices_jax][..., None] * chart_support_points_jax,
            axis=2,
        )
        jacobians = ein.contract(
            "nvi,qva->nqia",
            source_reference_points,
            shape_gradients_jax,
            backend="jax",
        )
        determinant_samples = jnp.linalg.det(jacobians).reshape((cells.shape[0], 3, 3, 3))
        determinant_controls = ein.contract(
            "ia,jb,kc,nabc->nijk",
            bernstein_inverse_jax,
            bernstein_inverse_jax,
            bernstein_inverse_jax,
            determinant_samples,
            backend="jax",
        )
        determinant_scales = jnp.max(
            jnp.abs(determinant_controls),
            axis=(1, 2, 3),
            keepdims=True,
        )
        normalized_controls = determinant_controls / determinant_scales
        chart_deficit = jnp.maximum(
            0.0,
            chart_positive_margin - normalized_controls,
        ).reshape((-1,))
        violations = jnp.concatenate(
            (scaled_deficit, mean_deficit, aspect_deficit, chart_deficit)
        )
        admissible = (
            jnp.all(quality.sampled_valid)
            & jnp.all(jnp.isfinite(violations))
            & jnp.all(determinant_scales > 0.0)
        )
        active_count = jnp.sum(violations > 0.0)
        aggregate = jnp.sum(violations**2) / jnp.maximum(active_count, 1)
        return jnp.where(admissible, aggregate, jnp.inf)

    initial_objective = float(np.asarray(quality_deficit(initial_parameters_jax)))
    if not np.isfinite(initial_objective):
        return False, "nonfinite_initial_quality_deficit"
    if initial_objective <= 0.0:
        return False, "quality_and_source_chart_bernstein_positive"
    objective_scale = jnp.asarray(initial_objective, dtype=jnp.float64)

    def root_support_energy(parameters: Any, _args: Any, /) -> Any:
        return quality_deficit(parameters) / objective_scale

    minimization = minimize(
        root_support_energy,
        initial_parameters_jax,
        method=ProjectedLBFGS(),
        termination=OptimizationTermination(maximum_steps=512),
        args=None,
        bounds=Bounds(),
    )
    optimized_parameters = np.asarray(minimization.parameters, dtype=np.float64)
    if optimized_parameters.shape != (len(initial_parameters),) or not np.all(
        np.isfinite(optimized_parameters)
    ):
        return False, "invalid_root_support_optimizer_parameters"
    logits = np.zeros((root_coordinates.shape[0], 4), dtype=np.float64)
    logits[np.asarray(parameter_rows), np.asarray(parameter_columns)] = (
        optimized_parameters
    )
    logits[~support_mask] = -np.inf
    probabilities = np.asarray(
        jax.nn.softmax(jnp.asarray(logits, dtype=jnp.float64), axis=1),
        dtype=np.float64,
    )
    updated_weights: list[Weights] = []
    updated_supports: list[tuple[int, ...]] = []
    updated_coordinates: list[Point3] = []
    for node_row, source_weights in enumerate(node_bank.weights):
        exact_probabilities = tuple(
            Fraction.from_float(float(probabilities[node_row, local]))
            for local in range(len(source_weights))
            if probabilities[node_row, local] > 0.0
        )
        active_vertices = tuple(
            vertex
            for local, (vertex, _) in enumerate(source_weights)
            if probabilities[node_row, local] > 0.0
        )
        total = sum(exact_probabilities, Fraction(0))
        if not total or not active_vertices:
            return False, "softmax_source_support_underflow"
        weights = tuple(
            (vertex, value / total)
            for vertex, value in zip(active_vertices, exact_probabilities, strict=True)
        )
        if sum((value for _, value in weights), Fraction(0)) != 1:
            return False, "non_partitioning_softmax_source_weights"
        root_point = tuple(
            sum(
                (weight * root_vertices[vertex][axis] for vertex, weight in weights),
                Fraction(0),
            )
            for axis in range(3)
        )
        updated_weights.append(weights)
        updated_supports.append(tuple(vertex for vertex, _ in weights))
        updated_coordinates.append(cast(Point3, root_point))
    if tuple(updated_weights) == tuple(node_bank.weights):
        return False, "root_support_optimizer_made_no_change"
    candidate_coordinates = np.asarray(
        [[float(value) for value in point] for point in updated_coordinates],
        dtype=np.float64,
    )
    candidate_mesh = root_mesh.with_coordinates(
        candidate_coordinates,
        numeric_version=source_mesh.numeric_version,
    )
    candidate_quality = evaluate_cell_quality(candidate_mesh)
    if not _cut_quality_meets_schedule(candidate_quality, schedule):
        candidate_aspect = np.asarray(candidate_quality.aspect_ratios, dtype=np.float64)
        return False, (
            "softmax_candidate_quality_failed:"
            f"scaled={float(np.min(np.asarray(candidate_quality.scaled_jacobian, dtype=np.float64))):.17g},"
            f"mean={float(np.min(np.asarray(candidate_quality.mean_ratios, dtype=np.float64))):.17g},"
            f"aspect={float(np.max(candidate_aspect)):.17g},"
            f"optimizer_status={int(np.asarray(minimization.status))},"
            f"optimizer_steps={int(np.asarray(minimization.diagnostics.iterations))},"
            f"optimizer_objective={float(np.asarray(minimization.objective)):.17g},"
            f"accepted_steps={int(np.asarray(minimization.diagnostics.accepted_steps))},"
            f"final_optimality={float(np.asarray(minimization.diagnostics.final_optimality_norm)):.17g}"
        )
    node_bank.weights[:] = updated_weights
    node_bank.supports[:] = updated_supports
    node_bank.coordinates[:] = updated_coordinates
    return True, "relocated"


def _compose_cut_hex_geometry(
    source: PiecewiseLinearComplex,
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    source_regions: np.ndarray,
    source_validity: CellValidityCertificate,
    plan: BalancedCutGridPlan,
    schedule: NativeHexGridSchedule,
    node_bank: _CutNodeBank,
    hexes: list[list[int]],
    parents: list[int],
    /,
) -> tuple[
    CellMesh,
    CellGeometrySpec,
    CellValidityCertificate,
    GlobalEmbeddingCertificate,
    _CutCoordinateMap,
    tuple[tuple[Fraction, Fraction], ...],
    tuple[Fraction, ...],
    str,
]:
    source_cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in source_mesh.blocks]
    )
    source_cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in source_mesh.blocks]
    )
    source_vertex_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
    source_expressions = _validate_cut_source_charts(
        source_mesh,
        source_geometry,
        plan,
        source_cells,
        tuple(
            _fraction_point3(point)
            for point in np.asarray(source_mesh.coordinates, dtype=np.float64)
        ),
        source_cell_ids,
    )
    _, relocation_status = _relocate_cut_nodes_in_root_coordinates(
        source_mesh,
        source_cells,
        source_expressions,
        node_bank,
        hexes,
        parents,
        schedule,
    )

    exact_vertices: list[ExactPoint3] = []
    coordinates = np.empty((len(node_bank.weights), 3), dtype=np.float64)
    for node_row, weights in enumerate(node_bank.weights):
        support = tuple(vertex for vertex, _ in weights)
        candidates = tuple(
            parent
            for parent, cell in enumerate(source_cells.tolist())
            if set(support) <= set(map(int, cell))
        )
        if not candidates:
            raise ValueError(f"Cut node {node_row} has no source tetrahedron ancestry.")
        images: tuple[ExactPoint3, ...] = tuple(
            (
                evaluate_expression(
                    source_expressions[parent][0],
                    _cut_source_reference_point(weights, parent, source_cells),
                ),
                evaluate_expression(
                    source_expressions[parent][1],
                    _cut_source_reference_point(weights, parent, source_cells),
                ),
                evaluate_expression(
                    source_expressions[parent][2],
                    _cut_source_reference_point(weights, parent, source_cells),
                ),
            )
            for parent in candidates
        )
        if any(image != images[0] for image in images[1:]):
            raise ValueError(
                f"Exact source charts disagree at shared cut-node row {node_row}."
            )
        exact_vertices.append(images[0])
        rounded = np.asarray([float(value) for value in images[0]], dtype=np.float64)
        if not np.all(np.isfinite(rounded)):
            raise ValueError("Exact curved cut coordinates must remain finite.")
        coordinates[node_row] = rounded

    source_elements, source_routes, _ = source_geometry.resolve(source_mesh)
    if len(source_elements) != 1:
        raise ValueError(
            "Exact curved cut-map composition requires one source coordinate block."
        )
    source_element = source_elements[0]
    if not isinstance(source_element, ScalarCoordinateElement):
        raise ValueError(
            "Exact curved cut-map composition requires a scalar source element."
        )
    source_route_table = np.asarray(source_routes[0], dtype=np.int64)
    root_elements: list[ScalarCoordinateElement] = []
    root_cell_ids: list[int] = []
    parent_to_root: list[tuple[Expression, Expression, Expression]] = []
    root_routes: list[tuple[int, ...]] = []
    for source_row, source_cell_id in enumerate(source_cell_ids.tolist()):
        root_element, root_cell_id, root_chart = _compose_restricted_dual_chart(
            source_element,
            source_row,
            int(source_cell_id),
            axes(3),
        )
        if len(root_chart) != 3:
            raise ValueError(
                "Cut source charts must have three exact reference coordinates."
            )
        root_elements.append(root_element)
        root_cell_ids.append(int(root_cell_id))
        parent_to_root.append((root_chart[0], root_chart[1], root_chart[2]))
        root_routes.append(tuple(int(value) for value in source_route_table[source_row]))
    if len({str(element.element_id) for element in root_elements}) != 1:
        raise ValueError(
            "Cut source cells with different exact root bases require separate output family blocks."
        )
    root_element = root_elements[0]

    charts: list[tuple[Expression, Expression, Expression]] = []
    chart_bounds: list[tuple[Fraction, Fraction]] = []
    reference_volumes: list[Fraction] = []
    for output_row, (cell, parent) in enumerate(zip(hexes, parents, strict=True)):
        source_references = tuple(
            _cut_source_reference_point(
                node_bank.weights[int(vertex)], int(parent), source_cells
            )
            for vertex in cell
        )
        chart = _cut_trilinear_chart(source_references)
        bounds = _cut_chart_bounds(chart, output_row)
        root_chart = tuple(
            compose_expression(argument, chart)
            for argument in parent_to_root[int(parent)]
        )
        if len(root_chart) != 3:
            raise ValueError(
                f"Curved cut chart for source cell ID {int(source_cell_ids[int(parent)])} "
                "must retain three exact root reference coordinates."
            )
        determinant = _cut_chart_determinant(chart)
        if isinstance(determinant, RationalPolynomial):
            raise ValueError("Cut chart coverage requires a polynomial determinant.")
        volume = Fraction(0)
        for exponent, coefficient in determinant.items():
            rational_coefficient = _require_fraction(
                coefficient,
                "Exact cut chart coverage requires rational polynomial coefficients.",
            )
            volume += rational_coefficient * Fraction(
                1,
                (exponent[0] + 1) * (exponent[1] + 1) * (exponent[2] + 1),
            )
        if volume <= 0:
            raise ValueError(
                f"Cut hexahedron chart {output_row} has nonpositive exact reference volume."
            )
        reference_volumes.append(volume)
        charts.append((root_chart[0], root_chart[1], root_chart[2]))
        chart_bounds.append(bounds)
    source_reference_coverage: list[Fraction] = [Fraction(0) for _ in source_cells]
    for parent, volume in zip(parents, reference_volumes, strict=True):
        source_reference_coverage[int(parent)] += volume
    for parent, volume in enumerate(source_reference_coverage):
        if volume != Fraction(1, 6):
            raise ValueError(
                f"Exact curved cut charts cover source cell ID "
                f"{int(source_cell_ids[parent])} with reference volume {volume}, not 1/6."
            )

    mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                "cells",
                "hexahedron",
                np.asarray(hexes, dtype=np.int64),
                global_ids=np.arange(len(hexes), dtype=np.int64),
            ),
        ),
        vertex_global_ids=fresh_vertex_global_ids(
            source_mesh, coordinates.shape[0] - source_mesh.coordinates.shape[0]
        ),
        numeric_version=source_mesh.numeric_version,
    )
    exact_vertex_tuple = tuple(exact_vertices)
    node_keys = tuple(node_bank.keys)
    exact_map = _CutCoordinateMap(
        plan.source_id,
        plan.source_revision,
        source,
        plan,
        schedule,
        source_mesh,
        source_geometry,
        source_regions,
        mesh,
        node_keys,
        exact_vertex_tuple,
    )
    source_parent_ids = source_cell_ids[np.asarray(parents, dtype=np.int64)]
    root_parent_ids = tuple(root_cell_ids[int(parent)] for parent in parents)
    source_parent_vertices = np.asarray(
        [source_vertex_ids[source_cells[int(parent)]] for parent in parents],
        dtype=np.int64,
    )
    reference_map: ReferenceCellMapWitnesses | None = None
    if (
        source_geometry.restriction_source is None
        and tuple(root_parent_ids)
        == tuple(int(value) for value in source_parent_ids.tolist())
        and all(
            str(block.cell_kind) == root_element.cell_kind for block in source_mesh.blocks
        )
    ):
        hexahedron_reference_vertices = tuple(
            tuple(Fraction.from_float(float(value)) for value in point)
            for point in np.asarray(
                reference_cell_topology("hexahedron").vertices,
                dtype=np.float64,
            )
        )
        exact_source_reference_vertices = tuple(
            tuple(
                tuple(
                    _require_fraction(
                        evaluate_expression(expression, point),
                        "Cut root-reference corners must remain rational.",
                    )
                    for expression in chart
                )
                for point in hexahedron_reference_vertices
            )
            for chart in charts
        )
        source_reference_vertices = np.asarray(
            exact_source_reference_vertices, dtype=np.float64
        )
        reference_map = ReferenceCellMapWitnesses(
            np.arange(len(charts), dtype=np.int64),
            np.asarray(root_parent_ids, dtype=np.int64),
            "hexahedron",
            root_element.cell_kind,
            source_reference_vertices,
            exact_source_reference_vertices,
            tuple(charts),
        )
    restriction = CellGeometryRestrictionSource(
        cell_geometry_id(source_geometry),
        source_mesh.topology_id,
        {"cells": source_parent_ids},
        {"cells": source_parent_vertices},
        block_source_blocks={"cells": source_mesh.blocks[0].name},
        block_parent_reference_maps={"cells": reference_map},
    )
    target_ids = tuple(int(value) for value in mesh.blocks[0].global_ids.tolist())
    routes = np.stack(
        [np.asarray(root_routes[int(parent)], dtype=np.int64) for parent in parents]
    )
    element = PerCellRestrictedCellGeometryElement(
        root_element,
        "hexahedron",
        target_ids,
        root_parent_ids,
        tuple(charts),
    )
    geometry = CellGeometrySpec(
        {"cells": element},
        {"cells": routes},
        source_geometry.coordinates,
        source_map=exact_map,
        source_mesh=mesh,
        restriction_source=restriction,
    )
    policy = CellValidityPolicy(
        relative_determinant_floor=0.0,
        relative_planarity_tolerance=source_validity.policy.relative_planarity_tolerance,
    )
    validity = certify_cell_geometry_validity(
        geometry,
        mesh=mesh,
        policy=policy,
    )
    status = np.asarray(validity.status)
    invalid_rows = tuple(
        int(row)
        for row in np.flatnonzero(status != int(CellValidityStatus.CERTIFIED_VALID))
    )
    if invalid_rows:
        invalid_output_ids = tuple(target_ids[row] for row in invalid_rows)
        invalid_root_ids = tuple(root_parent_ids[row] for row in invalid_rows)
        invalid_statuses = tuple(int(status[row]) for row in invalid_rows)
        invalid_determinant_bounds = tuple(
            (
                float(np.asarray(validity.determinant_lower[row])),
                float(np.asarray(validity.determinant_upper[row])),
            )
            for row in invalid_rows
        )
        raise ValueError(
            "Balanced cut-cell geometry lacks canonical positive whole-cell "
            "validity for output cell IDs "
            f"{invalid_output_ids}, root source cell IDs {invalid_root_ids}, "
            f"statuses {invalid_statuses}, determinant bounds {invalid_determinant_bounds}."
        )
    global_embedding = _require_global_embedding(
        mesh, geometry, validity, "Balanced cut-cell hexahedral output"
    )
    return (
        mesh,
        geometry,
        validity,
        global_embedding,
        exact_map,
        tuple(chart_bounds),
        tuple(source_reference_coverage),
        relocation_status,
    )


def _orient_hex_rows(
    cell_rows: list[int],
    exact_coordinates: Sequence[Point3],
    /,
) -> list[int]:
    """Orient from the exact carrier chart; whole-cell validity is certified later."""
    if len(cell_rows) != 8:
        raise ValueError("A cut-cell dual hexahedron must have eight corner rows.")
    reference = tuple(
        tuple(int(value) for value in point)
        for point in reference_cell_topology("hexahedron").vertices
    )
    center_derivatives: list[Point3] = []
    for axis in range(3):
        derivative_components = tuple(
            sum(
                (
                    Fraction(1 if bits[axis] else -1, 4)
                    * exact_coordinates[vertex_row][physical_axis]
                    for bits, vertex_row in zip(reference, cell_rows, strict=True)
                ),
                Fraction(0),
            )
            for physical_axis in range(3)
        )
        center_derivatives.append(
            (
                derivative_components[0],
                derivative_components[1],
                derivative_components[2],
            )
        )
    determinant = _determinant3(
        center_derivatives[0],
        center_derivatives[1],
        center_derivatives[2],
    )
    if determinant == 0:
        raise ValueError(
            "A cut-cell dual hexahedron has singular exact carrier orientation."
        )
    if determinant > 0:
        return cell_rows
    rows_by_bits = {bits: row for bits, row in zip(reference, cell_rows, strict=True)}
    return [rows_by_bits[(bits[0], bits[2], bits[1])] for bits in reference]


def _tetrahedral_coordinate_expressions(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    /,
) -> tuple[tuple[Expression, Expression, Expression], ...]:
    elements, routes, _ = source_geometry.resolve(source_mesh)
    expressions: list[tuple[Expression, Expression, Expression]] = []
    for block_index, block in enumerate(source_mesh.blocks):
        routes_host = np.asarray(routes[block_index], dtype=np.int64)
        source_ids = np.asarray(block.global_ids, dtype=np.int64)
        element = elements[block_index]
        if not isinstance(element, ScalarCoordinateElement):
            raise ValueError(
                "Exact cut charts require scalar source coordinate elements."
            )
        for row in range(block.cell_count):
            root_element, _, parent_to_root = _compose_restricted_dual_chart(
                element,
                row,
                int(source_ids[row]),
                axes(3),
            )
            coefficients = _exact_source_coordinate_bank(
                source_geometry.source_coordinates(routes_host[row])
            )
            basis = coordinate_effective_basis(root_element)
            if len(basis) != len(coefficients):
                raise ValueError(
                    "Root source geometry basis and route differ for exact cut composition."
                )
            components = tuple(
                tuple(coefficients[dof][axis] for dof in range(len(coefficients)))
                for axis in range(len(coefficients[0]))
            )
            root_expressions = linear_combinations(basis, components)
            chart = tuple(
                compose_expression(expression, parent_to_root)
                for expression in root_expressions
            )
            if len(chart) != 3:
                raise ValueError(
                    "Exact cut source coordinate charts must be three-dimensional."
                )
            expressions.append((chart[0], chart[1], chart[2]))
    return tuple(expressions)


def _tetrahedron_linear_component(
    values: tuple[Fraction, Fraction, Fraction, Fraction],
    /,
) -> Polynomial:
    polynomial = {
        (0, 0, 0): values[0],
        (1, 0, 0): values[1] - values[0],
        (0, 1, 0): values[2] - values[0],
        (0, 0, 1): values[3] - values[0],
    }
    return {
        exponent: coefficient
        for exponent, coefficient in polynomial.items()
        if coefficient != 0
    }


def _expression_nonnegative_on_tetrahedron(
    expression: Expression,
    /,
) -> bool:
    if isinstance(expression, RationalPolynomial):
        denominator = bernstein_coefficients(expression.denominator, "simplex", 3)
        if not denominator or not (
            all(value > 0 for value in denominator)
            or all(value < 0 for value in denominator)
        ):
            return False
        numerator = bernstein_coefficients(expression.numerator, "simplex", 3)
        if not numerator:
            return True
        return (
            all(value >= 0 for value in numerator)
            if all(value > 0 for value in denominator)
            else all(value <= 0 for value in numerator)
        )
    controls = bernstein_coefficients(expression, "simplex", 3)
    return bool(controls) and all(value >= 0 for value in controls)


def _expression_within_tetrahedron_range(
    expression: Expression,
    lower: Fraction,
    upper: Fraction,
    /,
) -> bool:
    if isinstance(expression, RationalPolynomial):
        denominator = bernstein_coefficients(expression.denominator, "simplex", 3)
        if not denominator or not (
            all(value > 0 for value in denominator)
            or all(value < 0 for value in denominator)
        ):
            return False
        denominator_positive = all(value > 0 for value in denominator)
        lower_numerator = add(expression.numerator, scale(expression.denominator, -lower))
        upper_numerator = add(
            scale(expression.denominator, upper),
            scale(expression.numerator, -1),
        )
        lower_controls = bernstein_coefficients(lower_numerator, "simplex", 3)
        upper_controls = bernstein_coefficients(upper_numerator, "simplex", 3)
        if not lower_controls or not upper_controls:
            return False
        if denominator_positive:
            return all(value >= 0 for value in lower_controls) and all(
                value >= 0 for value in upper_controls
            )
        return all(value <= 0 for value in lower_controls) and all(
            value <= 0 for value in upper_controls
        )
    lower_margin = add(expression, constant(-lower, 3))
    upper_margin = add(constant(upper, 3), scale(expression, -1))
    return _expression_nonnegative_on_tetrahedron(
        lower_margin
    ) and _expression_nonnegative_on_tetrahedron(upper_margin)


def _cut_source_face_chart(
    vertices: tuple[Point3, Point3, Point3],
    /,
) -> tuple[Polynomial, Polynomial, Polynomial]:
    return (
        {
            (0, 0): vertices[0][0],
            (1, 0): vertices[1][0] - vertices[0][0],
            (0, 1): vertices[2][0] - vertices[0][0],
        },
        {
            (0, 0): vertices[0][1],
            (1, 0): vertices[1][1] - vertices[0][1],
            (0, 1): vertices[2][1] - vertices[0][1],
        },
        {
            (0, 0): vertices[0][2],
            (1, 0): vertices[1][2] - vertices[0][2],
            (0, 1): vertices[2][2] - vertices[0][2],
        },
    )


def _is_tetrahedron_block(
    block: CellBlock | PolyhedralBlock,
    /,
) -> bool:
    if isinstance(block, PolyhedralBlock):
        return False
    topology = reference_cell_topology(block.cell_kind)
    return topology.dimension == 3 and topology.family == "tetrahedron"


def _tetrahedral_mesh_incidence(
    mesh: CellMesh,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    connectivity = mesh.connectivity
    if isinstance(connectivity, TetrahedralConnectivity):
        return (
            np.asarray(connectivity.edges, dtype=np.int64),
            np.asarray(connectivity.faces, dtype=np.int64),
            np.asarray(connectivity.face_cell_counts, dtype=np.int64),
        )
    if isinstance(connectivity, SimplicialConnectivity) and connectivity.dimension == 3:
        edges = np.asarray(connectivity.entities[1], dtype=np.int64)
        faces = np.asarray(connectivity.entities[2], dtype=np.int64)
        cell_faces = np.asarray(connectivity.cell_entities[2], dtype=np.int64)
        face_cell_counts = np.bincount(
            cell_faces.reshape((-1,)), minlength=faces.shape[0]
        ).astype(np.int64)
        return edges, faces, face_cell_counts
    raise ValueError(
        "Exact tetrahedral cut arrangements require three-dimensional simplicial connectivity."
    )


def _validate_cut_source_charts(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    plan: BalancedCutGridPlan,
    source_cells: np.ndarray,
    exact_points: tuple[Point3, ...],
    source_cell_ids: np.ndarray,
    /,
) -> tuple[tuple[Expression, Expression, Expression], ...]:
    expressions = _tetrahedral_coordinate_expressions(source_mesh, source_geometry)
    vertex_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
    _, source_faces, _ = _tetrahedral_mesh_incidence(source_mesh)
    source_face_ids = np.asarray(source_mesh.entity_set(2).entity_ids, dtype=np.int64)
    face_lookup: dict[tuple[int, int, int], int] = {
        _face_key3((int(face[0]), int(face[1]), int(face[2]))): int(source_face_ids[row])
        for row, face in enumerate(source_faces.tolist())
    }
    face_incidences: dict[
        tuple[int, int, int], list[tuple[int, tuple[int, int, int]]]
    ] = {}
    local_faces = reference_cell_topology("tetrahedron").entities[2]
    for parent, cell in enumerate(source_cells.tolist()):
        for local_face in local_faces:
            cycle = (
                int(cell[local_face[0]]),
                int(cell[local_face[1]]),
                int(cell[local_face[2]]),
            )
            face_incidences.setdefault(_face_key3(cycle), []).append((parent, cycle))
    for key, adjacent in face_incidences.items():
        if len(adjacent) > 2:
            raise ValueError(
                f"Curved source chart is undefined at nonmanifold source face ID {face_lookup[key]}."
            )
        if len(adjacent) != 2:
            continue
        canonical_ids = _face_key3(
            (
                int(vertex_ids[key[0]]),
                int(vertex_ids[key[1]]),
                int(vertex_ids[key[2]]),
            )
        )
        traces: list[tuple[Expression, Expression, Expression]] = []
        for parent, _ in adjacent:
            cell = tuple(int(value) for value in source_cells[parent])
            local_by_id: dict[int, Point3] = {
                int(vertex_ids[vertex]): _fraction_point3(
                    np.asarray(
                        reference_cell_topology("tetrahedron").vertices[local],
                        dtype=np.float64,
                    )
                )
                for local, vertex in enumerate(cell)
            }
            face_map = _cut_source_face_chart(
                (
                    local_by_id[canonical_ids[0]],
                    local_by_id[canonical_ids[1]],
                    local_by_id[canonical_ids[2]],
                )
            )
            trace = tuple(
                compose_expression(expression, face_map)
                for expression in expressions[parent]
            )
            traces.append((trace[0], trace[1], trace[2]))
        if any(
            not expression_equal(left, right)
            for left, right in zip(traces[0], traces[1], strict=True)
        ):
            raise ValueError(
                f"Exact curved source charts disagree across source face ID {face_lookup[key]}."
            )
    for cell_row, cell in enumerate(source_cells.tolist()):
        if plan.frame_layout_id is None:
            axis_knots = plan.axis_knots
            frame_axes = plan.frame_axes
            frame_origin = plan.frame_origin
        else:
            block = plan.frame_block_indices[cell_row]
            axis_knots = plan.block_axis_knots[block]
            frame_axes = plan.block_frame_axes[block]
            frame_origin = plan.block_frame_origins[block]
        points: tuple[Point3, Point3, Point3, Point3] = (
            exact_points[int(cell[0])],
            exact_points[int(cell[1])],
            exact_points[int(cell[2])],
            exact_points[int(cell[3])],
        )
        frame_points = tuple(
            _frame_point(point, frame_axes, frame_origin) for point in points
        )
        for axis in range(3):
            carrier_values = (
                frame_points[0][axis],
                frame_points[1][axis],
                frame_points[2][axis],
                frame_points[3][axis],
            )
            carrier_chart = _tetrahedron_linear_component(carrier_values)
            physical_chart = add(
                sum_polynomials(
                    tuple(
                        scale(
                            expressions[cell_row][coordinate],
                            frame_axes[axis][coordinate],
                        )
                        for coordinate in range(3)
                    ),
                    3,
                ),
                constant(-_dot3(frame_axes[axis], frame_origin), 3),
            )
            if len(axis_knots[axis]) > 2:
                if not expression_equal(physical_chart, carrier_chart):
                    raise ValueError(
                        f"Curved source cell ID {int(source_cell_ids[cell_row])} "
                        f"does not preserve the exact affine carrier component for "
                        f"cut axis {axis}; its internal grid-plane intersections "
                        f"have no linear root witness (source {physical_chart!r}, "
                        f"carrier {carrier_chart!r})."
                    )
            else:
                lower, upper = axis_knots[axis][0], axis_knots[axis][-1]
                if not _expression_within_tetrahedron_range(physical_chart, lower, upper):
                    raise ValueError(
                        f"Curved source cell ID {int(source_cell_ids[cell_row])} "
                        f"has no exact Bernstein range proof inside outer cut-axis "
                        f"{axis} knots."
                    )
                if not _expression_within_tetrahedron_range(carrier_chart, lower, upper):
                    raise ValueError(
                        f"Affine carrier cell ID {int(source_cell_ids[cell_row])} "
                        f"leaves the outer cut-axis {axis} grid interval."
                    )
    return expressions


@checked
def generate_balanced_cut_hexes(
    source: PiecewiseLinearComplex,
    source_mesh: CellMesh,
    cell_regions: ConvertibleToArray,
    plan: BalancedCutGridPlan,
    schedule: NativeHexGridSchedule,
    /,
    *,
    source_geometry: CellGeometrySpec | None = None,
) -> BalancedCutHexConstruction:
    """Clip in exact affine carrier axes; certify curved maps when cuts stay affine or uncut."""
    if not isinstance(source, PiecewiseLinearComplex) or not isinstance(
        source_mesh, CellMesh
    ):
        raise TypeError(
            "source and source_mesh must be PiecewiseLinearComplex and CellMesh."
        )
    if not isinstance(plan, BalancedCutGridPlan) or not isinstance(
        schedule, NativeHexGridSchedule
    ):
        raise TypeError(
            "plan and schedule must be BalancedCutGridPlan and NativeHexGridSchedule."
        )
    if (
        plan.source_complex_id != source.complex_id
        or plan.source_mesh_id != source_mesh.mesh_id
        or plan.source_topology_id != source_mesh.topology_id
        or schedule.route not in ("balanced_grid", "frame_grid")
        or (schedule.route == "frame_grid") != (plan.frame_id is not None)
        or (source.geometry is not None and source_geometry is None)
        or source.boundary != "conforming"
        or any(not _is_tetrahedron_block(block) for block in source_mesh.blocks)
    ):
        raise ValueError(
            "Cut-grid inputs must match one conforming PLC tetrahedral fill and schedule."
        )
    if source_geometry is None:
        source_geometry = CellGeometrySpec.affine(source_mesh)
    elif not isinstance(source_geometry, CellGeometrySpec):
        raise TypeError("source_geometry must be CellGeometrySpec or None.")
    regions = np.asarray(cell_regions)
    source_cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in source_mesh.blocks]
    )
    source_cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in source_mesh.blocks]
    )
    if regions.dtype.kind not in "iu" or regions.shape != (source_cells.shape[0],):
        raise ValueError(
            "cell_regions must provide one integer region per source tetrahedron."
        )
    region_rows = np.asarray(regions, dtype=np.int64)
    if np.any((region_rows < 0) | (region_rows >= len(source.region_ids))):
        raise ValueError("cell_regions must index declared PLC regions.")
    source_validity = certify_cell_geometry_validity(source_geometry, mesh=source_mesh)
    if source_validity.invalid_count or source_validity.unresolved_count:
        raise ValueError("Balanced cut input tetrahedra must be certified positive.")
    _require_tetrahedral_global_embedding(source_mesh, source_geometry, source_validity)
    edges, faces, face_incidence = _tetrahedral_mesh_incidence(source_mesh)
    if np.any(face_incidence > 2):
        bad_row = int(np.flatnonzero(face_incidence > 2)[0])
        face_id = int(source_mesh.entity_set(2).entity_ids[bad_row])
        raise ValueError(
            f"Cut-grid source tetrahedral fill is nonmanifold at source face ID {face_id}."
        )

    source_points = np.asarray(source_mesh.coordinates, dtype=np.float64)
    exact_points = tuple(_fraction_point3(point) for point in source_points)
    _validate_cut_source_charts(
        source_mesh,
        source_geometry,
        plan,
        source_cells,
        exact_points,
        source_cell_ids,
    )
    edge_ids = np.asarray(source_mesh.entity_set(1).entity_ids, dtype=np.int64)
    face_ids = np.asarray(source_mesh.entity_set(2).entity_ids, dtype=np.int64)
    edge_lookup: dict[tuple[int, int], int] = {
        _edge_key(int(edge[0]), int(edge[1])): row
        for row, edge in enumerate(edges.tolist())
    }
    face_lookup: dict[tuple[int, int, int], int] = {
        _face_key3((int(face[0]), int(face[1]), int(face[2]))): row
        for row, face in enumerate(faces.tolist())
    }
    source_face_row = face_lookup
    source_cell_rows = sorted(
        range(source_cells.shape[0]), key=lambda row: int(source_cell_ids[row])
    )
    clipped: list[_ClippedPolyhedron] = []
    clipped_volume_six = [Fraction(0) for _ in range(source_cells.shape[0])]
    source_volume_six: list[Fraction] = []
    for cell_row, cell in enumerate(source_cells.tolist()):
        corners: tuple[Point3, Point3, Point3, Point3] = (
            exact_points[int(cell[0])],
            exact_points[int(cell[1])],
            exact_points[int(cell[2])],
            exact_points[int(cell[3])],
        )
        source_volume_six.append(
            abs(
                _determinant3(
                    _subtract3(corners[1], corners[0]),
                    _subtract3(corners[2], corners[0]),
                    _subtract3(corners[3], corners[0]),
                )
            )
        )
    for cell_row in source_cell_rows:
        if plan.frame_layout_id is None:
            axis_knots = plan.axis_knots
            frame_axes = plan.frame_axes
            frame_origin = plan.frame_origin
        else:
            block = plan.frame_block_indices[cell_row]
            axis_knots = plan.block_axis_knots[block]
            frame_axes = plan.block_frame_axes[block]
            frame_origin = plan.block_frame_origins[block]
        cell_vertices = source_cells[cell_row]
        corners: tuple[Point3, Point3, Point3, Point3] = (
            exact_points[int(cell_vertices[0])],
            exact_points[int(cell_vertices[1])],
            exact_points[int(cell_vertices[2])],
            exact_points[int(cell_vertices[3])],
        )
        local_corners = tuple(
            _frame_point(point, frame_axes, frame_origin) for point in corners
        )
        ranges = []
        for axis in range(3):
            lower = min(point[axis] for point in local_corners)
            upper = max(point[axis] for point in local_corners)
            first = max(0, bisect_right(axis_knots[axis], lower) - 1)
            last = min(len(axis_knots[axis]) - 2, bisect_left(axis_knots[axis], upper))
            ranges.append(range(first, last + 1))
        source_planes = _tetrahedron_planes(
            source_cells,
            cell_row,
            faces,
            face_ids,
            source_face_row,
            exact_points,
        )
        for grid_index in product(*ranges):
            grid_cell = (
                int(grid_index[0]),
                int(grid_index[1]),
                int(grid_index[2]),
            )
            polyhedron = _clip_tetrahedron_to_grid_cell(
                source_mesh,
                source_cells,
                source_cell_ids,
                cell_row,
                grid_cell,
                plan,
                exact_points,
                source_planes,
                edge_lookup,
                face_lookup,
                edge_ids,
                face_ids,
                axis_knots=axis_knots,
                frame_axes=frame_axes,
                frame_origin=frame_origin,
            )
            if polyhedron is not None:
                clipped.append(polyhedron)
                clipped_volume_six[cell_row] += polyhedron.volume_six
    if not clipped:
        raise ValueError(
            "Exact background-cell clipping found no positive PLC intersections."
        )
    for cell_row, (covered, expected) in enumerate(
        zip(clipped_volume_six, source_volume_six, strict=True)
    ):
        if covered != expected:
            raise ValueError(
                "Exact grid clipping failed source-tetra coverage for cell ID "
                f"{int(source_cell_ids[cell_row])}: clipped six-volume {covered} "
                f"does not equal source six-volume {expected}."
            )

    source_clipped_polyhedra = tuple(clipped)
    used_tetrahedral_template = not _cut_polyhedra_are_trivalent(source_clipped_polyhedra)
    node_bank = _CutNodeBank(source_mesh)
    dual_polyhedra = (
        _tetrahedralize_cut_polyhedra(
            node_bank, source_clipped_polyhedra, source_cell_ids
        )
        if used_tetrahedral_template
        else source_clipped_polyhedra
    )
    (
        hexes,
        output_regions,
        parents,
        integer_cells,
        integer_blocks,
    ) = _dualize_cut_polyhedra_to_hexes(
        node_bank, dual_polyhedra, source_cell_ids, region_rows, plan
    )
    coordinates = np.asarray(
        [[float(value) for value in point] for point in node_bank.coordinates],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Rounded cut-cell coordinates must remain finite.")
    hexes = [_orient_hex_rows(row, node_bank.coordinates) for row in hexes]
    target_cells = np.asarray(hexes, dtype=np.int32)
    if target_cells.ndim != 2 or target_cells.shape[1] != 8:
        raise ValueError("Exact cut-polyhedron dualization produced no hexahedral cells.")
    if not used_tetrahedral_template:
        direct_mesh = CellMesh(
            coordinates,
            (
                CellBlock(
                    "cut-quality",
                    "hexahedron",
                    target_cells,
                    global_ids=np.arange(target_cells.shape[0], dtype=np.int64),
                ),
            ),
            vertex_global_ids=np.arange(coordinates.shape[0], dtype=np.int64),
            numeric_version=source_mesh.numeric_version,
        )
        direct_quality = evaluate_cell_quality(direct_mesh)
        if not np.all(np.asarray(direct_quality.sampled_valid, dtype=np.bool_)):
            # A negative sample rejects this topology; only the fully checked template
            # candidate below can be accepted.
            used_tetrahedral_template = True
            node_bank = _CutNodeBank(source_mesh)
            dual_polyhedra = _tetrahedralize_cut_polyhedra(
                node_bank, source_clipped_polyhedra, source_cell_ids
            )
            (
                hexes,
                output_regions,
                parents,
                integer_cells,
                integer_blocks,
            ) = _dualize_cut_polyhedra_to_hexes(
                node_bank, dual_polyhedra, source_cell_ids, region_rows, plan
            )
            coordinates = np.asarray(
                [[float(value) for value in point] for point in node_bank.coordinates],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(coordinates)):
                raise ValueError("Rounded cut-cell coordinates must remain finite.")
            hexes = [_orient_hex_rows(row, node_bank.coordinates) for row in hexes]
            target_cells = np.asarray(hexes, dtype=np.int32)
            if target_cells.ndim != 2 or target_cells.shape[1] != 8:
                raise ValueError(
                    "Exact tetrahedral-template dualization produced no hexahedral cells."
                )
    (
        mesh,
        geometry,
        validity,
        global_embedding,
        exact_map,
        chart_bounds,
        source_reference_coverage,
        relocation_status,
    ) = _compose_cut_hex_geometry(
        source,
        source_mesh,
        source_geometry,
        region_rows,
        source_validity,
        plan,
        schedule,
        node_bank,
        hexes,
        parents,
    )
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    quality = evaluate_cell_quality(mesh)
    sampled_valid = np.asarray(quality.sampled_valid, dtype=np.bool_)
    scaled = np.asarray(quality.scaled_jacobian, dtype=np.float64)
    mean = np.asarray(quality.mean_ratios, dtype=np.float64)
    aspect = np.asarray(quality.aspect_ratios, dtype=np.float64)
    if not np.all(sampled_valid):
        invalid_rows = tuple(int(row) for row in np.flatnonzero(~sampled_valid))
        invalid_cell_ids = tuple(
            int(mesh.blocks[0].global_ids[row]) for row in invalid_rows
        )
        raise ValueError(
            "Cut-cell quality observations found a nonpositive reference Jacobian at "
            f"output rows {invalid_rows} (cell IDs {invalid_cell_ids}); cut topology "
            f"used {'tetrahedral-template' if used_tetrahedral_template else 'direct-polyhedron-dual'}."
        )
    if (
        float(np.min(scaled)) < schedule.minimum_scaled_jacobian
        or float(np.min(mean)) < schedule.minimum_mean_ratio
        or float(np.max(aspect)) > schedule.maximum_aspect_ratio
    ):
        raise ValueError(
            "Balanced cut-cell output fails the declared whole-cell quality schedule: "
            f"after root-support relocation ({relocation_status}): "
            f"minimum scaled Jacobian {float(np.min(scaled)):.17g} "
            f"(required {schedule.minimum_scaled_jacobian:.17g}), minimum mean "
            f"ratio {float(np.min(mean)):.17g} (required "
            f"{schedule.minimum_mean_ratio:.17g}), maximum aspect ratio "
            f"{float(np.max(aspect)):.17g} (required at most "
            f"{schedule.maximum_aspect_ratio:.17g})."
        )
    parent_array = np.asarray(parents, dtype=np.int64)
    parent_offsets = np.arange(parent_array.size + 1, dtype=np.int64)
    parent_id_array = source_cell_ids[parent_array]
    region_array = np.asarray(output_regions, dtype=np.int64)
    integer_array = np.asarray(integer_cells, dtype=np.int64)
    integer_block_array = np.asarray(integer_blocks, dtype=np.int64)
    support_array = np.full((coordinates.shape[0], 4), -1, dtype=np.int64)
    for row, support in enumerate(node_bank.supports):
        if len(support) > 4:
            raise ValueError(
                "Cut-cell vertex support escapes its source tetrahedron ancestry."
            )
        support_array[row, : len(support)] = support
    vertex_key_tuple = tuple(node_bank.keys)
    construction_id = canonical_fingerprint(
        {
            "kind": "balanced-cut-cell-hex-construction",
            "complex": source.complex_id,
            "source_topology": source_mesh.topology_id,
            "plan": plan.plan_id,
            "source_geometry": cell_geometry_id(source_geometry),
            "source_validity": source_validity.certificate_id,
            "reference_chart_determinant_bounds": tuple(
                tuple((value.numerator, value.denominator) for value in bounds)
                for bounds in chart_bounds
            ),
            "source_reference_coverage": tuple(
                (value.numerator, value.denominator)
                for value in source_reference_coverage
            ),
            "mesh": mesh.mesh_id,
            "regions": region_array,
            "parents": parent_array,
            "source_region_ids": source.region_ids,
            "source_region_occurrence_paths": source.region_occurrence_paths,
            "source_plc_vertex_ids": source.vertex_ids,
            "source_plc_vertex_occurrence_paths": source.vertex_occurrence_paths,
            "source_plc_edge_ids": source.plc_edge_ids,
            "source_plc_edge_occurrence_paths": source.plc_edge_occurrence_paths,
            "source_plc_facet_ids": source.facet_ids,
            "source_plc_facet_occurrence_paths": source.facet_occurrence_paths,
            "parent_cell_ids": parent_id_array,
            "cut_vertex_keys": tuple(repr(key) for key in vertex_key_tuple),
            "integer_cells": integer_array,
            "integer_blocks": integer_block_array,
            "validity": validity.certificate_id,
            "global_embedding": global_embedding.certificate_id,
            "quality": {
                "scaled": scaled,
                "mean": mean,
                "aspect": aspect,
            },
        }
    )
    return BalancedCutHexConstruction(
        mesh,
        geometry,
        validity,
        global_embedding,
        region_array,
        parent_offsets,
        parent_array,
        parent_id_array,
        support_array,
        vertex_key_tuple,
        source.region_ids,
        source.region_occurrence_paths,
        source.vertex_ids,
        source.vertex_occurrence_paths,
        source.plc_edge_ids,
        source.plc_edge_occurrence_paths,
        source.facet_ids,
        source.facet_occurrence_paths,
        integer_array,
        integer_block_array,
        scaled,
        mean,
        aspect,
        source_mesh.mesh_id,
        source.complex_id,
        plan.plan_id,
        cell_geometry_id(source_geometry),
        source_validity.certificate_id,
        chart_bounds,
        source_reference_coverage,
        construction_id,
    )


__all__ = [
    "BalancedCutHexConstruction",
    "OrthogonalPlcHexConstruction",
    "generate_balanced_cut_hexes",
    "generate_orthogonal_plc_hexes",
]
