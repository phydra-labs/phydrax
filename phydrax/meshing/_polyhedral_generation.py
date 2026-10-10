#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Domain-restricted native power-cell generation on recovered PLC tetrahedra.

The tetrahedral carrier is constrained recovery, never an unconstrained convex
hull filtered by centroids. Cells join only within a material across unconstrained
faces. Non-star-visible components are decomposed into convex restricted pieces
for the existing degree-one VEM/FV consumers; their site/component lineage is
retained, not confused with a nodal interpolation map. Acceptance is independent
of this construction and belongs to ``providers._native_polyhedral``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from ..discretization._coordinate_enclosure import CoordinateEnclosureBudget

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import PLC_3D_COUNTERS, PlcRecovery3D, recover_plc_3d
from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometrySource,
)
from ..discretization._periodic_topology import (
    PeriodicIsometryGroup,
    PeriodicMeshTopology,
)
from ..geometry._mesh_certificates import PiecewiseLinearDomain
from ..geometry._triangulation import PeriodicPowerPreparation, RestrictedPowerDiagram
from ..typing import ConvertibleToArray
from ._controls import PeriodicConstraint
from ._quad_generation import _family_host_array
from ._volume_generation import _domain, PiecewiseLinearComplex


@dataclass(frozen=True, slots=True)
class NativePolyhedralSchedule:
    """Bounded site/weight refinement, with hard requests never silently relaxed.

    ``maximum_cell_diameter`` controls the actual restricted cell vertices, not
    the spacing of the sites. ``target_site_volumes`` sums every disconnected
    component of a site. Weight updates use the diagonal of the sparse power
    volume Jacobian, computed from reciprocal faces. Feature sites supplied in
    ``protected_sites`` are never relocated. Constrained curves additionally
    force decomposition so that recovery's feature edges remain mesh edges.
    """

    maximum_vertices: int = 1 << 22
    maximum_cells: int = 1 << 20
    maximum_work_units: int = 1 << 26
    maximum_scratch_bytes: int = 8_000_000_000
    maximum_refinement_steps: int = 8
    lloyd_steps: int = 0
    maximum_cell_diameter: float | None = None
    target_site_volumes: tuple[float, ...] = ()
    relative_volume_tolerance: float = 1e-6
    feature_tolerance: float = 1e-10
    protected_sites: tuple[int, ...] = ()
    require_every_site: bool = False

    def __post_init__(self) -> None:
        for name in (
            "maximum_vertices",
            "maximum_cells",
            "maximum_work_units",
            "maximum_scratch_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("maximum_refinement_steps", "lloyd_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        for name in ("relative_volume_tolerance", "feature_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.maximum_cell_diameter is not None and (
            not math.isfinite(self.maximum_cell_diameter)
            or self.maximum_cell_diameter <= 0
        ):
            raise ValueError("maximum_cell_diameter must be finite and positive.")
        if any(not math.isfinite(x) or x <= 0 for x in self.target_site_volumes):
            raise ValueError("Target site volumes must be finite and positive.")
        object.__setattr__(
            self,
            "target_site_volumes",
            tuple(float(value) for value in self.target_site_volumes),
        )
        object.__setattr__(self, "protected_sites", tuple(self.protected_sites))


@dataclass(frozen=True, slots=True)
class PolyhedralConstruction:
    mesh: CellMesh
    geometry: CellGeometrySpec
    domain: PiecewiseLinearDomain
    recovery: PlcRecovery3D
    diagram: RestrictedPowerDiagram
    cell_regions: np.ndarray
    cell_sites: np.ndarray
    face_facets: np.ndarray
    face_recovery_triangles: np.ndarray
    interface_faces: np.ndarray
    sheet_faces: np.ndarray
    feature_edge_sources: np.ndarray
    feature_source_edges: np.ndarray
    feature_maximum_distance: np.ndarray
    feature_coverage_gap: np.ndarray
    site_parents: np.ndarray
    component_ids: np.ndarray
    split_sites: np.ndarray
    requested_site_volumes: np.ndarray
    achieved_site_volumes: np.ndarray
    refinement_steps: int
    lloyd_steps: int
    optimization_relative_residual: float
    maximum_cell_diameter: float
    construction_id: str
    work_units: int
    periodic_constraints: tuple[PeriodicConstraint, ...] = ()
    seam_fragments: tuple[PeriodicPolyhedralSeamFragment, ...] = ()
    seam_face_pairs: np.ndarray | None = None
    seam_vertex_pairs: np.ndarray | None = None


def _polyhedral_periodic_group(
    constraints: tuple[PeriodicConstraint, ...],
    /,
) -> PeriodicIsometryGroup | None:
    if not constraints:
        return None
    if not all(isinstance(value, PeriodicConstraint) for value in constraints):
        raise TypeError("periodic_constraints must contain PeriodicConstraint values.")
    matrices = np.stack(
        [np.asarray(value.transform, dtype=np.float64) for value in constraints]
    )
    # This is the numerical group's classification bound, not a replacement
    # for ANY authored constraint. Original individual controls remain retained
    # and the source preparation/seam proof require exact original actions.
    return PeriodicIsometryGroup(
        matrices, tolerance=max(value.tolerance for value in constraints)
    )


class PolyhedralGenerationError(RuntimeError):
    """A bounded request remains unmet; actual entities and quantities survive."""

    def __init__(
        self, reason: str, *, entities: tuple[int, ...] = (), achieved: object = None
    ) -> None:
        self.reason, self.entities, self.achieved = reason, entities, achieved
        super().__init__(
            f"Native polyhedral generation: {reason}; entities={entities}; achieved={achieved}"
        )


@dataclass(frozen=True, slots=True)
class PeriodicPolyhedralSeamFragment:
    """One exact common-refinement polygon with both original parent actions.

    ``vertices`` are in the target image. Source coefficients act on the
    source triangle BEFORE the authored transform; target coefficients act
    directly on the target triangle. Neither action uses rounded vertices.
    """

    constraint_id: str
    source_face: int
    target_face: int
    source_triangle: tuple[int, int, int]
    target_triangle: tuple[int, int, int]
    vertices: tuple[tuple[Fraction, ...], ...]
    source_coefficients: tuple[tuple[Fraction, ...], ...]
    target_coefficients: tuple[tuple[Fraction, ...], ...]
    projected_measure: Fraction
    projection_axes: tuple[int, ...]
    relative_orientation: int


def _seam_triangle_actions(
    triangle: tuple[tuple[Fraction, ...], ...],
    vertices: tuple[tuple[Fraction, ...], ...],
    axes: tuple[int, ...],
    ledger: CoordinateEnclosureBudget,
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    from ..geometry._planar_coverage import _reserve_fraction_work
    from ..linalg._small_batched import prepare_exact_small_linear_actions

    ledger.reserve(0, 512 + 48 * len(vertices))
    matrix = (
        (Fraction(1),) * 3,
        tuple(point[axes[0]] for point in triangle),
        tuple(point[axes[1]] for point in triangle),
    )
    right = (
        (Fraction(1),) * len(vertices),
        tuple(point[axes[0]] for point in vertices),
        tuple(point[axes[1]] for point in vertices),
    )
    solved = prepare_exact_small_linear_actions(matrix, right, coordinate_budget=ledger)
    if solved.actions is None:
        raise PolyhedralGenerationError("periodic_seam_source_rank")
    actions = tuple(
        tuple(row[column] for row in solved.actions) for column in range(len(vertices))
    )
    _reserve_fraction_work((*triangle, *vertices), 40 * len(vertices), len(vertices), 6)
    for coefficients, point in zip(actions, vertices, strict=True):
        if any(value < 0 for value in coefficients) or sum(coefficients) != 1:
            raise PolyhedralGenerationError("periodic_seam_source_containment")
        if any(
            sum(coefficients[index] * triangle[index][axis] for index in range(3))
            != point[axis]
            for axis in range(3)
        ):
            raise PolyhedralGenerationError("periodic_seam_source_action")
    return actions


@dataclass(frozen=True, slots=True)
class _PeriodicSeamTriangle:
    face: int
    corners: tuple[int, int, int]
    points: tuple[tuple[Fraction, ...], ...]
    plane: tuple[Fraction, ...]
    axes: tuple[int, ...]
    projected: tuple[tuple[Fraction, ...], ...]
    measure: Fraction


def _seam_parent_triangles(
    coordinates: tuple[tuple[Fraction, ...], ...],
    point_array: np.ndarray,
    faces: np.ndarray,
    offsets: np.ndarray,
    rows: np.ndarray,
    matrix: tuple[tuple[Fraction, ...], ...] | None,
    ledger: CoordinateEnclosureBudget,
    /,
) -> tuple[_PeriodicSeamTriangle, ...]:
    from ..geometry._exact_polyhedral_geometry import (
        triangulate_loop,
        triangulation_charge,
    )
    from ..geometry._planar_coverage import (
        _reserve_fraction_work,
        plane_key,
        project,
        signed_measure,
    )

    result = []
    for face in faces:
        a, b = offsets[face : face + 2]
        loop = tuple(map(int, rows[a:b]))
        work, storage = triangulation_charge(point_array, loop)
        ledger.reserve(work, storage)
        for triangle in triangulate_loop(point_array, loop):
            points = tuple(coordinates[index] for index in triangle)
            if matrix is not None:
                _reserve_fraction_work(points, 36, 9, 8)
                points = tuple(
                    tuple(
                        sum(matrix[axis][column] * point[column] for column in range(3))
                        + matrix[axis][3]
                        for axis in range(3)
                    )
                    for point in points
                )
            key = plane_key(points)
            if key is None:
                raise PolyhedralGenerationError(
                    "periodic_seam_degenerate_triangle", entities=(int(face),)
                )
            projected = project(points, key[1])
            ledger.reserve(1, 512)
            result.append(
                _PeriodicSeamTriangle(
                    int(face),
                    triangle,
                    points,
                    key[0],
                    key[1],
                    projected,
                    abs(signed_measure(projected)),
                )
            )
    return tuple(result)


def _seam_pair_fragment(
    source: _PeriodicSeamTriangle,
    target: _PeriodicSeamTriangle,
    constraint_id: str,
    ledger: CoordinateEnclosureBudget,
    /,
) -> PeriodicPolyhedralSeamFragment | None:
    from ..geometry._planar_coverage import (
        _reserve_fraction_work,
        intersection_polygon,
        signed_measure,
    )

    ledger.reserve(1)
    if source.plane != target.plane:
        return None
    raw_polygon = intersection_polygon(target.projected, source.projected)
    # Compact only EXACT duplicate constructed intersections, never source
    # coordinates or tolerance-near vertices. Keep the oriented clipping order.
    ledger.reserve(len(raw_polygon), 256 + 16 * len(raw_polygon))
    polygon = tuple(
        point
        for index, point in enumerate(raw_polygon)
        if point != raw_polygon[index - 1]
    )
    measure = abs(signed_measure(polygon))
    if not measure:
        return None
    axes, plane = target.axes, target.plane
    missing_axis = next(axis for axis in range(3) if axis not in axes)
    _reserve_fraction_work(
        (*source.points, *target.points, *polygon), 6 * len(polygon), 3 * len(polygon), 12
    )
    vertices = []
    for point in polygon:
        row = [Fraction(0)] * 3
        row[axes[0]], row[axes[1]] = point
        row[missing_axis] = (
            -(plane[3] + sum(plane[axis] * row[axis] for axis in axes))
            / plane[missing_axis]
        )
        vertices.append(tuple(row))
    bank = tuple(vertices)
    source_actions = _seam_triangle_actions(source.points, bank, axes, ledger)
    target_actions = _seam_triangle_actions(target.points, bank, axes, ledger)
    orientation = (
        1
        if signed_measure(source.projected) * signed_measure(target.projected) > 0
        else -1
    )
    fragment = PeriodicPolyhedralSeamFragment(
        constraint_id,
        source.face,
        target.face,
        source.corners,
        target.corners,
        bank,
        source_actions,
        target_actions,
        measure,
        axes,
        orientation,
    )
    ledger.retain_basis(
        (
            fragment,
            constraint_id,
            source.face,
            target.face,
            source.corners,
            target.corners,
            bank,
            source_actions,
            target_actions,
            measure,
            axes,
            orientation,
        )
    )
    return fragment


def _seam_group_fragments(
    source: tuple[_PeriodicSeamTriangle, ...],
    target: tuple[_PeriodicSeamTriangle, ...],
    constraint_id: str,
    ledger: CoordinateEnclosureBudget,
    /,
) -> tuple[PeriodicPolyhedralSeamFragment, ...]:
    from ..geometry._planar_coverage import _reserve_fraction_work, intersection

    for triangles in (source, target):
        for index, first in enumerate(triangles):
            for second_index in range(index + 1, len(triangles)):
                second = triangles[second_index]
                ledger.reserve(1)
                if (
                    first.plane == second.plane
                    and intersection(first.projected, second.projected) > 0
                ):
                    raise PolyhedralGenerationError(
                        "periodic_seam_double_coverage",
                        entities=(first.face, second.face),
                    )
    ledger.reserve(0, 256 + 16 * (len(source) + len(target)))
    covered_source = [Fraction(0) for _ in source]
    covered_target = [Fraction(0) for _ in target]
    result = []
    for source_index, first in enumerate(source):
        for target_index, second in enumerate(target):
            fragment = _seam_pair_fragment(first, second, constraint_id, ledger)
            if fragment is not None:
                ledger.reserve(1, 16)
                _reserve_fraction_work(
                    (
                        (
                            covered_source[source_index],
                            covered_target[target_index],
                            fragment.projected_measure,
                        ),
                    ),
                    2,
                    2,
                    2,
                )
                result.append(fragment)
                covered_source[source_index] += fragment.projected_measure
                covered_target[target_index] += fragment.projected_measure
    for triangles, covered in ((source, covered_source), (target, covered_target)):
        bad = tuple(
            triangle.face
            for triangle, measure in zip(triangles, covered, strict=True)
            if triangle.measure != measure
        )
        if bad:
            raise PolyhedralGenerationError("periodic_seam_exact_coverage", entities=bad)
    ledger.reserve(0, 256 + 8 * len(result))
    return tuple(result)


def prepare_periodic_polyhedral_seam_fragments(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    face_facets: np.ndarray,
    constraints: Sequence[PeriodicConstraint],
    /,
    *,
    maximum_work: int,
    maximum_scratch_bytes: int,
) -> tuple[PeriodicPolyhedralSeamFragment, ...]:
    """Exact reciprocal seam common refinement of the actual source geometry.

    Scientific facet scopes, not coordinate proximity, select parent faces.
    Canonical source triangulation and clipping handle unequal subdivisions
    and concave parent loops. Every selected triangle must be covered exactly
    once on BOTH sides; an authored tolerance is never source normalization.
    """
    from ..discretization._coordinate_enclosure import (
        coordinate_enclosure_budget,
        prepared_coordinate_source_bank,
    )

    c = _polyhedral_connectivity(mesh)
    facets = np.asarray(face_facets)
    if facets.shape != (c.face_count,) or not np.issubdtype(facets.dtype, np.integer):
        raise ValueError(
            "face_facets must name every published face's scientific source facet."
        )
    values = tuple(constraints)
    if not all(isinstance(value, PeriodicConstraint) for value in values):
        raise TypeError("constraints must contain PeriodicConstraint values.")
    offsets, rows = np.asarray(c.face_vertex_offsets), np.asarray(c.face_vertex_values)
    ledger = coordinate_enclosure_budget(maximum_work, maximum_scratch_bytes)
    fragments: list[PeriodicPolyhedralSeamFragment] = []
    with ledger.activate(), ledger.bound_stage(maximum_work, maximum_scratch_bytes):
        coordinates = prepared_coordinate_source_bank(geometry)
        geometry._resolve(mesh, exact_source_prepared=True)
        ledger.reserve(0, 256 + 24 * len(coordinates))
        point_array = np.asarray(coordinates, dtype=object)
        for constraint in values:
            if (
                constraint.source_scope.entity_dimension != 2
                or constraint.target_scope.entity_dimension != 2
            ):
                raise PolyhedralGenerationError("periodic_seam_facet_scope")
            source_facets = tuple(
                map(int, np.asarray(constraint.source_scope.entity_ids))
            )
            target_facets = tuple(
                map(int, np.asarray(constraint.target_scope.entity_ids))
            )
            if constraint.source_entity_ids is None:
                groups = ((source_facets, target_facets),)
            else:
                groups = tuple(
                    ((int(source),), (target,))
                    for source, target in zip(
                        np.asarray(constraint.source_entity_ids),
                        target_facets,
                        strict=True,
                    )
                )
            matrix = tuple(
                tuple(Fraction(float(value)) for value in row)
                for row in np.asarray(constraint.transform)
            )
            if len(matrix) != 4:
                raise PolyhedralGenerationError("periodic_seam_transform_dimension")
            for source_group, target_group in groups:
                source_faces = np.flatnonzero(np.isin(facets, source_group))
                target_faces = np.flatnonzero(np.isin(facets, target_group))
                if not source_faces.size or not target_faces.size:
                    raise PolyhedralGenerationError(
                        "periodic_seam_missing_facet",
                        entities=(*source_group, *target_group),
                    )
                source = _seam_parent_triangles(
                    coordinates, point_array, source_faces, offsets, rows, matrix, ledger
                )
                target = _seam_parent_triangles(
                    coordinates, point_array, target_faces, offsets, rows, None, ledger
                )
                group = _seam_group_fragments(
                    source, target, constraint.constraint_id, ledger
                )
                ledger.reserve(len(group), 256 + 16 * len(group))
                fragments.extend(group)
        ledger.reserve(0, 256 + 8 * len(fragments))
        result = tuple(fragments)
        ledger.retain_basis((result,))
    return result


@dataclass(frozen=True, slots=True)
class _PeriodicSeamBank:
    points: tuple[tuple[Fraction, ...], ...]
    parents: np.ndarray
    coefficients: tuple[tuple[Fraction, ...], ...]
    actions: np.ndarray
    carriers: np.ndarray
    face_parts: dict[int, tuple[tuple[int, ...], ...]]
    paired_loops: tuple[tuple[tuple[int, ...], tuple[int, ...], int], ...]
    vertex_pairs: np.ndarray


def _periodic_seam_bank(
    source: ExactPowerCellGeometrySource,
    diagram: RestrictedPowerDiagram,
    fragments: tuple[PeriodicPolyhedralSeamFragment, ...],
    constraints: tuple[PeriodicConstraint, ...],
    maximum_vertices: int,
    ledger: CoordinateEnclosureBudget,
    /,
) -> _PeriodicSeamBank:
    from ..geometry._planar_coverage import _reserve_fraction_work

    preparation = diagram.periodic_preparation
    if preparation is None:
        raise ValueError(
            "Periodic seam assembly requires its original power preparation."
        )
    prepared = source.prepare()
    points: list[tuple[Fraction, ...]] = list(prepared.vertices)
    rank = len(preparation.orders)
    ledger.reserve(len(points), 512 + 256 * len(points))
    parents = [(index, -1, -1) for index in range(len(points))]
    coefficients: list[tuple[Fraction, ...]] = [
        (Fraction(1), Fraction(0), Fraction(0))
    ] * len(points)
    actions = [(0,) * rank] * len(points)
    carriers = []
    carrier_rows = diagram.construction.vertex_carriers
    for vertex_index, weights in enumerate(prepared.barycentric):
        indices = tuple(
            int(carrier_rows[vertex_index, column])
            for column in range(carrier_rows.shape[1])
            if carrier_rows[vertex_index, column] >= 0
        )
        carriers.append(
            tuple(
                index
                for index, weight in zip(indices, weights, strict=True)
                if weight > 0
            )
        )
    lookup = {}
    for index, point in enumerate(points):
        lookup.setdefault(point, index)
    constraint_indices = {
        value.constraint_id: index for index, value in enumerate(constraints)
    }
    parts: dict[int, list[tuple[int, ...]]] = {}
    paired_loops = []
    vertex_pairs = []

    def vertex(
        point: tuple[Fraction, ...],
        row: tuple[int, int, int],
        weights: tuple[Fraction, ...],
        exponents: tuple[int, ...],
        physical_row: tuple[int, int, int],
        physical_weights: tuple[Fraction, ...],
    ) -> int:
        if point in lookup:
            return lookup[point]
        if len(points) >= maximum_vertices:
            raise PolyhedralGenerationError(
                "periodic_vertex_budget", achieved=len(points) + 1
            )
        ledger.reserve(1, 512)
        index = len(points)
        points.append(point)
        parents.append(row)
        coefficients.append(weights)
        actions.append(exponents)
        support = {
            carrier
            for parent, weight in zip(physical_row, physical_weights, strict=True)
            if weight
            for carrier in carriers[parent]
        }
        carriers.append(tuple(sorted(support)))
        lookup[point] = index
        ledger.retain_basis((point, row, weights, exponents, carriers[-1]))
        return index

    for fragment in fragments:
        generator = constraint_indices[fragment.constraint_id]
        exponents = tuple(int(axis == generator) for axis in range(rank))
        if array_tree_fingerprint(
            preparation.generators[generator]
        ) != array_tree_fingerprint(np.asarray(constraints[generator].transform)):
            raise ValueError(
                "Seam constraint actions differ from their original periodic preparation."
            )
        left, right = [], []
        for point, source_weights, target_weights in zip(
            fragment.vertices,
            fragment.source_coefficients,
            fragment.target_coefficients,
            strict=True,
        ):
            triangle = tuple(
                prepared.vertices[index] for index in fragment.source_triangle
            )
            _reserve_fraction_work(triangle, 18, 3, 6)
            original = tuple(
                sum(
                    (
                        weight * parent[axis]
                        for parent, weight in zip(triangle, source_weights, strict=True)
                    ),
                    Fraction(0),
                )
                for axis in range(3)
            )
            source_vertex = vertex(
                original,
                fragment.source_triangle,
                source_weights,
                (0,) * rank,
                fragment.source_triangle,
                source_weights,
            )
            target_vertex = vertex(
                point,
                fragment.source_triangle,
                source_weights,
                exponents,
                fragment.target_triangle,
                target_weights,
            )
            left.append(source_vertex)
            right.append(target_vertex)
            ledger.reserve(1, 128)
            vertex_pairs.append((source_vertex, target_vertex, generator))
        source_loop = tuple(left if fragment.relative_orientation > 0 else reversed(left))
        target_loop = tuple(right)
        parts.setdefault(fragment.source_face, []).append(source_loop)
        parts.setdefault(fragment.target_face, []).append(target_loop)
        paired_loops.append((source_loop, target_loop, generator))
    count = len(points)
    parent_bank = _family_host_array((count, 3), np.int64)
    action_bank = _family_host_array((count, rank), np.int64)
    carrier_bank = _family_host_array(
        (count, max(4, max(map(len, carriers), default=0))), np.int32
    )
    carrier_bank.fill(-1)
    for index in range(count):
        parent_bank[index] = parents[index]
        action_bank[index] = actions[index]
        carrier_bank[index, : len(carriers[index])] = carriers[index]
    pairs = _family_host_array((len(vertex_pairs), 3), np.int64)
    for index, pair in enumerate(vertex_pairs):
        pairs[index] = pair
    return _PeriodicSeamBank(
        tuple(points),
        parent_bank,
        tuple(coefficients),
        action_bank,
        carrier_bank,
        {face: tuple(loops) for face, loops in parts.items()},
        tuple(paired_loops),
        pairs,
    )


def _seam_segment_subdivision(
    points: tuple[tuple[Fraction, ...], ...],
    loops: Sequence[tuple[int, ...]],
    ledger: CoordinateEnclosureBudget,
    /,
) -> dict[tuple[int, int], tuple[int, ...]]:
    from ..geometry._planar_coverage import _reserve_fraction_work
    from ..linalg._small_batched import prepare_exact_small_linear_actions

    edges = {
        tuple(sorted((first, second)))
        for loop in loops
        for first, second in zip(loop, (*loop[1:], loop[0]), strict=True)
    }
    result = {}
    for first, second in sorted(edges):
        start, stop = points[first], points[second]
        _reserve_fraction_work((start, stop), 3, 3, 2)
        direction = tuple(b - a for a, b in zip(start, stop, strict=True))
        pivot = next((axis for axis, value in enumerate(direction) if value), None)
        if pivot is None:
            raise PolyhedralGenerationError(
                "periodic_seam_degenerate_edge", entities=(first, second)
            )
        interior = []
        for index, point in enumerate(points):
            if index in (first, second):
                continue
            with ledger.temporary_scope():
                _reserve_fraction_work((point, start), 1, 1, 2)
                solve = prepare_exact_small_linear_actions(
                    ((direction[pivot],),),
                    ((point[pivot] - start[pivot],),),
                    coordinate_budget=ledger,
                )
                if solve.actions is None:
                    raise PolyhedralGenerationError("periodic_seam_edge_rank")
                parameter = solve.actions[0][0]
                _reserve_fraction_work((start, direction, point), 9, 3, 4)
                inside = 0 < parameter < 1 and all(
                    start[axis] + parameter * direction[axis] == point[axis]
                    for axis in range(3)
                )
            if inside:
                ledger.retain_basis((parameter, index))
                ledger.reserve(1, 128)
                interior.append((parameter, index))
        ledger.reserve(
            len(interior) * max(1, len(interior).bit_length()), 256 + 16 * len(interior)
        )
        result[(first, second)] = (
            first,
            *(index for _, index in sorted(interior)),
            second,
        )
    return result


def _expanded_seam_loop(
    loop: tuple[int, ...],
    edges: dict[tuple[int, int], tuple[int, ...]],
    /,
) -> tuple[int, ...]:
    result = []
    for first, second in zip(loop, (*loop[1:], loop[0]), strict=True):
        edge = edges[(min(first, second), max(first, second))]
        result.extend(edge[:-1] if first < second else edge[:0:-1])
    return tuple(result)


def _seam_vertex_orbits(
    points: tuple[tuple[Fraction, ...], ...],
    pairs: np.ndarray,
    identifiers: np.ndarray,
    preparation: PeriodicPowerPreparation,
    ledger: CoordinateEnclosureBudget,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    from ..discretization._periodic_topology import _exact_periodic_element
    from ..geometry._planar_coverage import _reserve_fraction_work

    matrices = tuple(
        tuple(tuple(Fraction(float(value)) for value in row) for row in matrix)
        for matrix in preparation.generators
    )
    orders = preparation.orders
    rank = len(orders)
    parent = list(range(len(points)))
    winding = [(0,) * rank for _ in points]

    def normalized(values: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            value % order if order else value
            for value, order in zip(values, orders, strict=True)
        )

    def root(vertex: int) -> tuple[int, tuple[int, ...]]:
        total = (0,) * rank
        while parent[vertex] != vertex:
            ledger.reserve(rank)
            total = normalized(
                tuple(a + b for a, b in zip(total, winding[vertex], strict=True))
            )
            vertex = parent[vertex]
        return vertex, total

    for source, target, generator in pairs:
        source_root, source_shift = root(int(source))
        target_root, target_shift = root(int(target))
        difference = normalized(
            tuple(
                int(axis == generator) + a - b
                for axis, (a, b) in enumerate(
                    zip(source_shift, target_shift, strict=True)
                )
            )
        )
        if source_root == target_root:
            matrix = _exact_periodic_element(matrices, orders, difference)
            point = points[source_root]
            _reserve_fraction_work((point, *matrix), 36, 3, 8)
            image = tuple(
                sum(matrix[axis][column] * point[column] for column in range(3))
                + matrix[axis][3]
                for axis in range(3)
            )
            if image != point:
                raise PolyhedralGenerationError(
                    "periodic_seam_orbit_cycle", entities=(int(source), int(target))
                )
        elif identifiers[source_root] < identifiers[target_root]:
            parent[target_root], winding[target_root] = source_root, difference
        else:
            parent[source_root] = target_root
            winding[source_root] = normalized(tuple(-value for value in difference))
    representatives = _family_host_array((len(points),), np.int64)
    shifts = _family_host_array((len(points), rank), np.int64)
    for index in range(len(points)):
        representative, values = root(index)
        if any(
            value < np.iinfo(np.int32).min or value > np.iinfo(np.int32).max
            for value in values
        ):
            raise PolyhedralGenerationError(
                "periodic_seam_winding_budget", entities=(index,)
            )
        representatives[index], shifts[index] = representative, values
    return representatives, shifts


@dataclass(frozen=True, slots=True)
class _PeriodicPolyhedralAssembly:
    mesh: CellMesh
    geometry: CellGeometrySpec
    parent_faces: np.ndarray
    vertex_carriers: np.ndarray
    face_pairs: np.ndarray
    vertex_pairs: np.ndarray


def _reconcile_periodic_polyhedral(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    diagram: RestrictedPowerDiagram,
    fragments: tuple[PeriodicPolyhedralSeamFragment, ...],
    constraints: tuple[PeriodicConstraint, ...],
    group: PeriodicIsometryGroup,
    maximum_vertices: int,
    ledger: CoordinateEnclosureBudget,
    /,
) -> _PeriodicPolyhedralAssembly:
    source = geometry.exact_source
    if (
        not isinstance(source, ExactPowerCellGeometrySource)
        or diagram.periodic_preparation is None
    ):
        raise ValueError(
            "Periodic reconciliation requires its actual original exact power source."
        )
    old = _polyhedral_connectivity(mesh)
    bank = _periodic_seam_bank(
        source, diagram, fragments, constraints, maximum_vertices, ledger
    )
    offsets, vertices = (
        np.asarray(old.face_vertex_offsets),
        np.asarray(old.face_vertex_values),
    )
    parts = {
        face: bank.face_parts.get(face, (tuple(map(int, vertices[a:b])),))
        for face, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True))
    }
    all_loops = tuple(loop for loops in parts.values() for loop in loops)
    subdivision = _seam_segment_subdivision(bank.points, all_loops, ledger)
    expanded = {
        face: tuple(_expanded_seam_loop(loop, subdivision) for loop in loops)
        for face, loops in parts.items()
    }
    origins = {
        tuple(sorted(loop)): face for face, loops in expanded.items() for loop in loops
    }
    cell_offsets, cell_faces = (
        np.asarray(old.cell_face_offsets),
        np.asarray(old.cell_face_values),
    )
    signs = np.asarray(old.cell_face_sign_values)
    cells = []
    for a, b in zip(cell_offsets[:-1], cell_offsets[1:], strict=True):
        cells.append(
            [
                np.asarray(loop if sign > 0 else loop[::-1], dtype=np.int64)
                for face, sign in zip(cell_faces[a:b], signs[a:b], strict=True)
                for loop in expanded[int(face)]
            ]
        )
    old_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    identifiers = _family_host_array((len(bank.points),), np.int64)
    identifiers[: len(old_ids)] = old_ids
    first = int(np.max(old_ids)) + 1
    if first + len(bank.points) - len(old_ids) > np.iinfo(np.int64).max:
        raise PolyhedralGenerationError("periodic_vertex_identity_budget")
    identifiers[len(old_ids) :] = np.arange(
        first, first + len(bank.points) - len(old_ids), dtype=np.int64
    )
    linear = ExactPowerCellGeometryLinearActionSource(
        source,
        bank.parents,
        bank.coefficients,
        bank.actions,
        periodic_preparation=diagram.periodic_preparation,
    )
    prepared = linear.prepare()
    if prepared.vertices != bank.points:
        raise PolyhedralGenerationError("periodic_seam_source_binding")
    lifted = CellMesh.from_polyhedra(
        prepared.rounded_vertices,
        cells,
        block_name="restricted-power",
        numeric_version=mesh.numeric_version,
        vertex_global_ids=identifiers,
        cell_global_ids=np.asarray(old.cell_global_ids),
    )
    lifted_geometry = CellGeometrySpec.power(lifted, linear)
    representatives, shifts = _seam_vertex_orbits(
        bank.points, bank.vertex_pairs, identifiers, diagram.periodic_preparation, ledger
    )
    topology = PeriodicMeshTopology(
        lifted, group, representatives, shifts, actual_geometry=lifted_geometry
    )
    bound = CellMesh(
        lifted.coordinates,
        lifted.blocks,
        vertex_global_ids=lifted.vertex_global_ids,
        polyhedral_connectivity=_polyhedral_connectivity(lifted),
        periodic_topology=topology,
        numeric_version=lifted.numeric_version,
    )
    bound_geometry = CellGeometrySpec.power(bound, linear)
    connectivity = _polyhedral_connectivity(bound)
    new_offsets, new_vertices = (
        np.asarray(connectivity.face_vertex_offsets),
        np.asarray(connectivity.face_vertex_values),
    )
    lookup = {
        tuple(sorted(map(int, new_vertices[a:b]))): face
        for face, (a, b) in enumerate(zip(new_offsets[:-1], new_offsets[1:], strict=True))
    }
    parent_faces = _family_host_array((connectivity.face_count,), np.int64)
    for key, face in lookup.items():
        parent_faces[face] = origins[key]
    face_pairs = _family_host_array((len(bank.paired_loops), 3), np.int64)
    for index, (left, right, generator) in enumerate(bank.paired_loops):
        face_pairs[index] = (
            lookup[tuple(sorted(_expanded_seam_loop(left, subdivision)))],
            lookup[tuple(sorted(_expanded_seam_loop(right, subdivision)))],
            generator,
        )
    return _PeriodicPolyhedralAssembly(
        bound, bound_geometry, parent_faces, bank.carriers, face_pairs, bank.vertex_pairs
    )


def _freeze(value: ConvertibleToArray, /) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _polyhedral_connectivity(mesh: CellMesh, /) -> PolyhedralConnectivity:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("This route requires canonical packed polyhedral connectivity.")
    return connectivity


def _tet_facets(recovery: PlcRecovery3D) -> np.ndarray:
    # Facet ids are semantic groups and may span noncoplanar polygons. Merge
    # constrained fragments only on the same planar recovered subtriangle.
    faces = {
        tuple(sorted(map(int, face))): index for index, face in enumerate(recovery.faces)
    }
    return np.asarray(
        [
            [
                faces.get(tuple(sorted(int(tet[v]) for v in range(4) if v != k)), -1)
                for k in range(4)
            ]
            for tet in recovery.tetrahedra
        ],
        dtype=np.int32,
    )


def _publish(
    diagram: RestrictedPowerDiagram,
    recovery: PlcRecovery3D,
    revision: str,
    maximum_work: int,
    *,
    domain_source: PiecewiseLinearComplex,
    periodic_constraints: tuple[PeriodicConstraint, ...],
) -> tuple[CellMesh, CellGeometrySpec, np.ndarray, np.ndarray]:
    data = diagram.construction
    source = ExactPowerCellGeometrySource(
        diagram.points,
        diagram.weights,
        recovery.points,
        recovery.tetrahedra,
        data.vertex_site_offsets,
        data.vertex_sites,
        data.vertex_carriers,
        maximum_work=maximum_work,
        periodic_preparation=diagram.periodic_preparation,
        domain_source=domain_source,
        periodic_constraints=periodic_constraints,
    )
    exact = source.prepare()
    cells: list[list[np.ndarray]] = [[] for _ in data.cell_sites]
    for face, (left, right) in enumerate(
        zip(data.face_cells[:, 0], data.face_cells[:, 1], strict=True)
    ):
        start, stop = data.face_offsets[face : face + 2]
        loop = data.face_vertices[start:stop]
        cells[int(left)].append(loop)
        if right >= 0:
            cells[int(right)].append(loop[::-1])
    mesh = CellMesh.from_polyhedra(
        exact.rounded_vertices,
        cells,
        block_name="restricted-power",
        numeric_version=revision,
    )
    geometry = CellGeometrySpec.power(mesh, source)
    # CellMesh groups arities, retaining input cell ids. Reorder all ancestry by
    # those ids rather than assuming publication preserves native row order.
    order = np.asarray(_polyhedral_connectivity(mesh).cell_global_ids, dtype=np.int64)
    native_faces = {
        tuple(sorted(map(int, data.face_vertices[a:b]))): index
        for index, (a, b) in enumerate(
            zip(data.face_offsets[:-1], data.face_offsets[1:], strict=True)
        )
    }
    connectivity = _polyhedral_connectivity(mesh)
    offsets, values = (
        np.asarray(connectivity.face_vertex_offsets),
        np.asarray(connectivity.face_vertex_values),
    )
    face_order = np.asarray(
        [
            native_faces[tuple(sorted(map(int, values[a:b])))]
            for a, b in zip(offsets[:-1], offsets[1:], strict=True)
        ],
        dtype=np.int64,
    )
    return mesh, geometry, order, face_order


def _diameters(mesh: CellMesh, geometry: CellGeometrySpec) -> np.ndarray:
    # Twice the radius about the exact vertex mean bounds the ideal diameter.
    c = _polyhedral_connectivity(mesh)
    points = np.asarray(geometry.source_coordinates(), dtype=object)
    offsets, values = np.asarray(c.cell_vertex_offsets), np.asarray(c.cell_vertex_values)
    bounds = []
    for a, b in zip(offsets[:-1], offsets[1:], strict=True):
        vertices = tuple(tuple(points[index]) for index in values[a:b])
        center = tuple(
            sum((point[axis] for point in vertices), Fraction(0)) / len(vertices)
            for axis in range(3)
        )
        squared_radius = max(
            sum(((point[axis] - center[axis]) ** 2 for axis in range(3)), Fraction(0))
            for point in vertices
        )
        bounds.append(_sqrt_upper(4 * squared_radius))
    return np.asarray(bounds, dtype=np.float64)


def _centroid_in_kernel(
    mesh: CellMesh, geometry: CellGeometrySpec, cell: int, point: np.ndarray
) -> bool:
    from ..geometry._planar_coverage import plane_key

    c = _polyhedral_connectivity(mesh)
    face_offsets, faces = np.asarray(c.cell_face_offsets), np.asarray(c.cell_face_values)
    signs = np.asarray(c.cell_face_sign_values)
    offsets, vertices = (
        np.asarray(c.face_vertex_offsets),
        np.asarray(c.face_vertex_values),
    )
    coordinates = np.asarray(geometry.source_coordinates(), dtype=object)
    query = tuple(Fraction(float(value)) for value in point)
    start, stop = face_offsets[cell : cell + 2]
    for face, sign in zip(faces[start:stop], signs[start:stop], strict=True):
        a, b = offsets[face : face + 2]
        loop = vertices[a:b]
        if sign < 0:
            loop = loop[::-1]
        polygon = tuple(tuple(coordinates[index]) for index in loop)
        key = next(
            (
                plane_key((polygon[0], polygon[index], polygon[index + 1]))
                for index in range(1, len(polygon) - 1)
                if plane_key((polygon[0], polygon[index], polygon[index + 1])) is not None
            ),
            None,
        )
        if (
            key is None
            or key[3] * sum((key[0][axis] * query[axis] for axis in range(3)), key[0][3])
            >= 0
        ):
            return False
    return True


def _sqrt_upper(value: Fraction) -> float:
    from fractions import Fraction

    if value == 0:
        return 0.0
    if float(value) == 0:
        raise PolyhedralGenerationError("feature_measure_unrepresentable")
    result = float(np.nextafter(math.sqrt(float(value)), np.inf))
    if not math.isfinite(result):
        raise PolyhedralGenerationError("feature_measure_unrepresentable")
    while Fraction(result) ** 2 < value:
        result = float(np.nextafter(result, np.inf))
    return result


def _segment_evidence(
    ends: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[Fraction, Fraction, float, float]:
    from fractions import Fraction

    origin = tuple(x if isinstance(x, Fraction) else Fraction(float(x)) for x in a)
    direction = tuple(
        (y if isinstance(y, Fraction) else Fraction(float(y))) - x
        for x, y in zip(origin, b, strict=True)
    )
    squared_length = sum((x * x for x in direction), Fraction(0))
    parameters: list[Fraction] = []
    squared_distances: list[Fraction] = []
    for end in ends:
        delta = tuple(
            (x if isinstance(x, Fraction) else Fraction(float(x))) - y
            for x, y in zip(end, origin, strict=True)
        )
        parameter = (
            sum((x * y for x, y in zip(delta, direction, strict=True)), Fraction(0))
            / squared_length
        )
        parameters.append(parameter)
        nearest = min(Fraction(1), max(Fraction(0), parameter))
        squared_distances.append(
            sum(
                ((x - nearest * y) ** 2 for x, y in zip(delta, direction, strict=True)),
                Fraction(0),
            )
        )
    return (
        max(Fraction(0), min(parameters)),
        min(Fraction(1), max(parameters)),
        _sqrt_upper(max(squared_distances)),
        _sqrt_upper(squared_length),
    )


def _source_feature_edges(
    complex_: PiecewiseLinearComplex, recovery: PlcRecovery3D
) -> np.ndarray:
    """Preserve semantic curves AND geometric creases within one facet group."""
    from .._meshcore import exact_orient3d

    triangles: dict[int, np.ndarray] = {}
    for triangle, polygon in zip(
        recovery.input_triangles, recovery.input_polygons, strict=True
    ):
        triangles.setdefault(
            int(polygon), np.asarray(complex_.vertices[triangle], dtype=np.float64)
        )
    incidence: dict[tuple[int, ...], list[int]] = {}
    offsets = complex_.polygon_offsets
    for polygon, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        loop = complex_.polygon_vertices[a:b]
        for first, second in zip(loop, np.roll(loop, -1), strict=True):
            incidence.setdefault(tuple(sorted((int(first), int(second)))), []).append(
                polygon
            )
    edges = [tuple(map(int, pair)) for pair in recovery.plc_edges]
    known = {tuple(sorted(pair)) for pair in edges}
    for edge, polygons in sorted(incidence.items()):
        if edge in known:
            continue
        facets = {int(complex_.polygon_facets[polygon]) for polygon in polygons}
        sharp = len(facets) > 1 or len(polygons) == 1
        anchor = triangles[polygons[0]]
        for polygon in polygons[1:]:
            other = triangles[polygon]
            signs = exact_orient3d(
                np.broadcast_to(anchor[0], other.shape),
                np.broadcast_to(anchor[1], other.shape),
                np.broadcast_to(anchor[2], other.shape),
                other,
            )
            sharp = sharp or bool(np.any(signs != 0))
        if sharp:
            edges.append(edge)
    return np.asarray(edges, dtype=np.int32).reshape((-1, 2))


def _feature_ancestry(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    recovery: PlcRecovery3D,
    diagram: RestrictedPowerDiagram,
    source_edges: np.ndarray,
    work_limit: int,
    *,
    vertex_carriers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Carrier ancestry plus exact affine feature-distance/coverage bounds.

    Geometry alone cannot distinguish nearby thin features. An output edge
    first names its tetrahedral carrier edge; recovered segment ancestry
    identifies that edge's source. Unregistered crease subedges are accepted
    only on an EXACT source segment, using a bounded BVH candidate search.
    Ideal output coordinates are independently measured through the canonical
    source coefficient bank; RNE carrier equality is not a source segment proof.
    """
    from fractions import Fraction

    from .._bvh import bvh_overlap_pair_blocks, prepare_bvh

    edges = np.asarray(_polyhedral_connectivity(mesh).edges, dtype=np.int64)
    points = np.asarray(geometry.source_coordinates(), dtype=object)
    carriers = (
        diagram.construction.vertex_carriers
        if vertex_carriers is None
        else vertex_carriers
    )
    sources = np.full(edges.shape[0], -1, dtype=np.int64)
    distances = np.zeros(source_edges.shape[0])
    gaps = np.zeros_like(distances)
    if source_edges.shape[0] == 0:
        return sources, distances, gaps, 0
    carrier_sources = {
        tuple(sorted(map(int, pair))): source for source, pair in enumerate(source_edges)
    }
    for pair, source in zip(recovery.segments, recovery.segment_sources, strict=True):
        carrier_sources[tuple(sorted(map(int, pair)))] = int(source)
    source_points = recovery.points[source_edges]
    tree = prepare_bvh(
        np.min(source_points, axis=1), np.max(source_points, axis=1), dtype=np.float64
    )
    intervals = [[] for _ in source_edges]
    lengths = [_segment_evidence(pair, pair[0], pair[1])[3] for pair in source_points]
    work = 0
    for edge, pair in enumerate(edges):
        work += 1
        if work > work_limit:
            raise PolyhedralGenerationError(
                "feature_candidate_budget", entities=(edge,), achieved=work
            )
        key = tuple(
            sorted({int(point) for point in carriers[pair].reshape(-1) if point >= 0})
        )
        if len(key) != 2:
            continue
        if key not in carrier_sources:
            carrier_points = recovery.points[list(key)]
            carrier_tree = prepare_bvh(
                np.min(carrier_points, axis=0)[None, :],
                np.max(carrier_points, axis=0)[None, :],
                dtype=np.float64,
            )
            matched = []
            for candidates, _ in bvh_overlap_pair_blocks(
                tree, carrier_tree, include_touching=True
            ):
                for source in candidates:
                    work += 1
                    if work > work_limit:
                        raise PolyhedralGenerationError(
                            "feature_candidate_budget", entities=(edge,), achieved=work
                        )
                    a, b = source_points[source]
                    start, stop, residual, _ = _segment_evidence(carrier_points, a, b)
                    if residual == 0.0 and stop > start:
                        matched.append(int(source))
            if len(matched) > 1:
                raise PolyhedralGenerationError(
                    "ambiguous_feature_ancestry", entities=(edge,)
                )
            carrier_sources[key] = matched[0] if matched else -1
        source = carrier_sources[key]
        if source < 0:
            continue
        sources[edge] = source
        a, b = source_points[source]
        start, stop, residual, _ = _segment_evidence(points[pair], a, b)
        distances[source] = max(distances[source], residual)
        if stop > start:
            intervals[source].append((start, stop))
    missing = tuple(source for source, pieces in enumerate(intervals) if not pieces)
    if missing:
        raise PolyhedralGenerationError(
            "missing_feature_edge", entities=missing, achieved={"covered_segments": 0.0}
        )
    for source, pieces in enumerate(intervals):
        cursor, gap = Fraction(0), Fraction(0)
        for start, stop in sorted(pieces):
            gap = max(gap, start - cursor)
            cursor = max(cursor, stop)
        gap = max(gap, Fraction(1) - cursor) * Fraction(lengths[source])
        upper_gap = float(gap)
        if Fraction(upper_gap) < gap:
            upper_gap = float(np.nextafter(upper_gap, np.inf))
        gaps[source] = upper_gap
    return sources, distances, gaps, work


def _numerical_power_sites(
    diagram: RestrictedPowerDiagram,
    ledger: CoordinateEnclosureBudget,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """RNE optimization view of retained images; never a geometry authority."""
    from ..geometry._planar_coverage import _reserve_fraction_work, rational_points

    preparation = diagram.periodic_preparation
    if preparation is None:
        return diagram.points, np.arange(len(diagram.points), dtype=np.int64)
    points = _family_host_array((len(preparation.image_sites), 3), np.float64)
    offsets, components = (
        preparation.image_coordinate_offsets,
        preparation.image_coordinate_components,
    )
    for coordinate, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        with ledger.temporary_scope():
            bank = rational_points(components[a:b].reshape((-1, 1)))
            _reserve_fraction_work(bank, len(bank) + 1, 1, max(1, len(bank)))
            value = sum((row[0] for row in bank), Fraction(0))
            points[coordinate // 3, coordinate % 3] = float(value)
    return points, preparation.image_sites


def generate_polyhedral_volume(
    complex_: PiecewiseLinearComplex,
    /,
    *,
    sites: object | None = None,
    weights: object | None = None,
    schedule: NativePolyhedralSchedule | None = None,
    source_id: str = "native-polyhedral",
    seeds: object | None = None,
    seed_regions: object | None = None,
    protected_vertices: tuple[int, ...] = (),
    periodic_constraints: Sequence[PeriodicConstraint] = (),
    record_native_phase: Callable[[str, float, int | None, int], None] | None = None,
) -> PolyhedralConstruction:
    """Generate arbitrary nonconvex/material PLC-restricted power polyhedra.

    No accepted status is assigned here. The provider separately certifies the
    actual coordinates, global embedding and declared material-domain coverage.
    """
    if not isinstance(complex_, PiecewiseLinearComplex):
        raise TypeError("complex_ must be PiecewiseLinearComplex.")
    policy = NativePolyhedralSchedule() if schedule is None else schedule
    if not isinstance(policy, NativePolyhedralSchedule):
        raise TypeError("schedule must be NativePolyhedralSchedule.")
    periodic_values = tuple(periodic_constraints)
    periodic_group = _polyhedral_periodic_group(periodic_values)
    recovery = recover_plc_3d(
        complex_.vertices,
        complex_.polygon_offsets,
        complex_.polygon_vertices,
        complex_.polygon_facets,
        complex_.facet_regions,
        segments=complex_.segments,
        seeds=seeds,
        seed_regions=seed_regions,
        boundary_policy=complex_.boundary,
        max_vertices=policy.maximum_vertices,
        max_tetrahedra=policy.maximum_cells,
        work_limit=policy.maximum_work_units,
        max_scratch_bytes=policy.maximum_scratch_bytes,
    )
    domain = _domain(
        complex_, recovery.input_triangles, recovery.input_polygons, source_id
    )
    facets = _tet_facets(recovery)
    # The default singleton diagram is subsequently refined under an explicit
    # size request; it is not a disguised dense background sampler.
    site_array = (
        np.mean(recovery.points[recovery.tetrahedra[0]], axis=0)[None, :]
        if sites is None
        else np.asarray(sites, dtype=np.float64)
    )
    if (
        site_array.ndim != 2
        or site_array.shape[1] != 3
        or site_array.shape[0] < 1
        or not np.all(np.isfinite(site_array))
    ):
        raise ValueError("sites must be a finite nonempty (n, 3) array.")
    site_array = site_array.copy()
    values = (
        np.zeros(site_array.shape[0])
        if weights is None
        else np.asarray(weights, dtype=np.float64).copy()
    )
    if values.shape != (site_array.shape[0],) or not np.all(np.isfinite(values)):
        raise ValueError("weights must be a finite (site_count,) array.")
    protected = set(policy.protected_sites)
    if any(
        isinstance(x, bool) or not isinstance(x, int) or x < 0 or x >= site_array.shape[0]
        for x in protected
    ):
        raise ValueError("protected_sites must name existing sites.")
    targets = np.asarray(policy.target_site_volumes, dtype=np.float64)
    if targets.size and targets.shape != values.shape:
        raise ValueError("target_site_volumes must match the supplied sites.")
    if targets.size and np.unique(site_array, axis=0).shape[0] != site_array.shape[0]:
        raise PolyhedralGenerationError("coincident_positive_volume_targets")
    preserve_vertices = bool(protected_vertices)
    if any(
        vertex < 0 or vertex >= complex_.vertices.shape[0]
        for vertex in protected_vertices
    ):
        raise ValueError("protected_vertices must name PLC input vertices.")
    split = np.full(
        site_array.shape[0],
        bool(complex_.segments.shape[0]) or preserve_vertices,
        dtype=np.int8,
    )
    parents = np.arange(site_array.shape[0], dtype=np.int64)
    refinement_steps = 0
    lloyd_steps = 0
    spent_work = int(recovery.counters[PLC_3D_COUNTERS.index("work_units")])
    residual = 0.0
    while True:
        if spent_work >= policy.maximum_work_units:
            raise PolyhedralGenerationError("work_budget", achieved=spent_work)
        diagram = RestrictedPowerDiagram(
            site_array,
            values,
            recovery.points,
            recovery.tetrahedra,
            recovery.tetrahedron_regions,
            facets,
            split_sites=split,
            max_pieces=policy.maximum_cells,
            max_vertices=policy.maximum_vertices,
            work_limit=policy.maximum_work_units - spent_work,
            record_native_phase=record_native_phase,
            periodic_group=periodic_group,
            maximum_images=policy.maximum_work_units - spent_work
            if periodic_group is not None
            else None,
        )
        spent_work += int(
            diagram.construction.counters[0] + diagram.construction.counters[3]
        )
        spent_work += diagram.preparation_work_units
        spent_work += diagram.adjacency_work_units
        spent_work += diagram.construction.source_work_units
        if spent_work >= policy.maximum_work_units:
            raise PolyhedralGenerationError("work_budget", achieved=spent_work)
        from ..discretization._coordinate_enclosure import CoordinateEnclosureBudget

        exact_ledger = CoordinateEnclosureBudget(
            policy.maximum_work_units - spent_work, policy.maximum_scratch_bytes
        )
        with exact_ledger.activate():
            mesh, geometry, order, face_order = _publish(
                diagram,
                recovery,
                domain.source_revision,
                policy.maximum_work_units - spent_work,
                domain_source=complex_,
                periodic_constraints=periodic_values,
            )
            validity = certify_cell_geometry_validity(geometry, mesh=mesh)
        spent_work += exact_ledger.work_units
        unresolved = np.asarray(validity.status) != int(
            CellValidityStatus.CERTIFIED_VALID
        )
        if np.any(unresolved):
            offending = np.unique(diagram.cell_original_sites[order[unresolved]])
            if np.any(split[offending]):
                raise PolyhedralGenerationError(
                    "piece_geometry_unresolved",
                    entities=tuple(map(int, order[unresolved])),
                )
            split[offending] = 1
            continue
        from ..discretization._cell_geometry_transfer import _certified_cell_measures

        measure_ledger = CoordinateEnclosureBudget(
            policy.maximum_work_units - spent_work, policy.maximum_scratch_bytes
        )
        with measure_ledger.activate():
            exact_volumes, _, _ = _certified_cell_measures(
                mesh, geometry, maximum_work=policy.maximum_work_units - spent_work
            )
        spent_work += measure_ledger.work_units
        achieved = np.bincount(
            diagram.cell_original_sites[order],
            weights=exact_volumes,
            minlength=site_array.shape[0],
        )
        empty = np.flatnonzero(achieved == 0)
        if policy.require_every_site and empty.size and not targets.size:
            raise PolyhedralGenerationError(
                "required_empty_sites", entities=tuple(map(int, empty))
            )
        diameters = _diameters(mesh, geometry)
        large = (
            np.flatnonzero(diameters > policy.maximum_cell_diameter)
            if policy.maximum_cell_diameter is not None
            else np.zeros(0, dtype=np.int64)
        )
        residual = (
            float(np.max(np.abs(achieved - targets) / targets)) if targets.size else 0.0
        )
        needs_weights = targets.size and residual > policy.relative_volume_tolerance
        if not large.size and not needs_weights and lloyd_steps >= policy.lloyd_steps:
            if policy.require_every_site and empty.size:
                raise PolyhedralGenerationError(
                    "required_empty_sites", entities=tuple(map(int, empty))
                )
            break
        if refinement_steps >= policy.maximum_refinement_steps:
            raise PolyhedralGenerationError(
                "refinement_unresolved",
                entities=tuple(map(int, order[large])),
                achieved={
                    "diameter_upper": float(np.max(diameters)),
                    "relative_volume_residual": residual,
                },
            )
        if needs_weights:
            # Sparse diagonal Jacobian: dV_i/dw_i = sum A_ij/(2|s_i-s_j|).
            diagonal = np.zeros_like(values)
            data = diagram.construction
            image_ledger = CoordinateEnclosureBudget(
                policy.maximum_work_units - spent_work, policy.maximum_scratch_bytes
            )
            with image_ledger.activate():
                metric_sites, metric_owners = _numerical_power_sites(
                    diagram, image_ledger
                )
                image_ledger.charge_native_work(
                    image_ledger.work_units - image_ledger.native_charged_work_units
                )
            spent_work += image_ledger.work_units
            for f, (left, right) in enumerate(
                zip(data.face_cells[:, 0], data.face_cells[:, 1], strict=True)
            ):
                if right < 0:
                    continue
                i, j = (
                    int(diagram.cell_original_sites[left]),
                    int(diagram.cell_original_sites[right]),
                )
                if i == j:
                    continue
                a, b = data.face_offsets[f : f + 2]
                polygon = data.vertices[data.face_vertices[a:b]]
                area_vector = (
                    np.sum(
                        np.cross(
                            polygon - polygon[0],
                            np.roll(polygon, -1, axis=0) - polygon[0],
                        ),
                        axis=0,
                    )
                    / 2
                )
                derivative = np.linalg.norm(area_vector) / (
                    2
                    * np.linalg.norm(
                        metric_sites[data.cell_sites[left]]
                        - metric_sites[data.cell_sites[right]]
                    )
                )
                diagonal[i] += derivative
                diagonal[j] += derivative
            if empty.size:
                # Revive hidden weighted sites by a sparse power-distance walk
                # at an interior carrier point, not an all-site distance table.
                point = np.mean(recovery.points[recovery.tetrahedra[0]], axis=0)
                owner = int(data.cell_sites[0])
                best = float(
                    np.sum((point - metric_sites[owner]) ** 2)
                    - values[metric_owners[owner]]
                )
                while True:
                    next_owner, next_best = owner, best
                    a, b = diagram.neighbor_offsets[owner : owner + 2]
                    for neighbor in diagram.neighbors[a:b]:
                        if neighbor >= len(metric_sites):
                            continue
                        candidate = float(
                            np.sum((point - metric_sites[neighbor]) ** 2)
                            - values[metric_owners[neighbor]]
                        )
                        if candidate < next_best:
                            next_owner, next_best = int(neighbor), candidate
                    if next_owner == owner:
                        break
                    owner, best = next_owner, next_best
                for site in empty:
                    images = np.flatnonzero(metric_owners == site)
                    distance = min(
                        float(np.sum((point - metric_sites[image]) ** 2))
                        for image in images
                    )
                    values[site] = float(
                        distance - best + 0.1 * targets[site] ** (2.0 / 3.0)
                    )
                refinement_steps += 1
                continue
            if np.any(diagonal <= 0):
                raise PolyhedralGenerationError(
                    "weight_jacobian_unresolved",
                    entities=tuple(map(int, np.flatnonzero(diagonal <= 0))),
                )
            values += 0.5 * (targets - achieved) / diagonal
            values -= values[0]
        if (
            large.size
            and targets.size
            and not needs_weights
            and lloyd_steps >= policy.lloyd_steps
        ):
            raise PolyhedralGenerationError(
                "fixed_site_diameter_unresolved",
                entities=tuple(map(int, order[large])),
                achieved={"cell_diameter_upper": float(np.max(diameters))},
            )
        if large.size and not targets.size:
            c = _polyhedral_connectivity(mesh)
            offsets, vertices = (
                np.asarray(c.cell_vertex_offsets),
                np.asarray(c.cell_vertex_values),
            )
            centers = np.asarray(
                [
                    np.mean(
                        np.asarray(mesh.coordinates)[
                            vertices[offsets[k] : offsets[k + 1]]
                        ],
                        axis=0,
                    )
                    for k in large
                ]
            )
            # A certified vertex-mean star center lies inside the actual cell.
            new_sites = np.concatenate((site_array, centers))
            if np.unique(new_sites, axis=0).shape[0] != new_sites.shape[0]:
                raise PolyhedralGenerationError(
                    "site_insertion_stalled", entities=tuple(map(int, order[large]))
                )
            source_sites = diagram.cell_original_sites[order[large]]
            site_array = new_sites
            values = np.concatenate((values, values[source_sites]))
            parents = np.concatenate((parents, source_sites))
            split = np.concatenate(
                (
                    split,
                    np.full(
                        large.size,
                        bool(complex_.segments.shape[0]) or preserve_vertices,
                        dtype=np.int8,
                    ),
                )
            )
        if lloyd_steps < policy.lloyd_steps:
            # Restrict relocation to single unsplit star-visible components;
            # averaging disconnected components can move a site into a hole.
            data = diagram.construction
            multiplicity = np.bincount(
                diagram.cell_original_sites, minlength=achieved.size
            )
            inverse_order = np.empty_like(order)
            inverse_order[order] = np.arange(order.size)
            for cell, site in enumerate(diagram.cell_original_sites):
                if (
                    int(site) not in protected
                    and multiplicity[site] == 1
                    and not split[site]
                ):
                    centroid = data.cell_moments[cell] / data.cell_volumes[cell]
                    if _centroid_in_kernel(
                        mesh, geometry, int(inverse_order[cell]), centroid
                    ):
                        site_array[site] = centroid
            lloyd_steps += 1
        refinement_steps += 1
    data = diagram.construction
    regions = data.cell_regions[order]
    face_recovery_triangles = data.face_facets[face_order]
    face_facets = np.full(face_recovery_triangles.size, -1, dtype=np.int32)
    constrained_triangles = face_recovery_triangles >= 0
    face_facets[constrained_triangles] = recovery.face_sources[
        face_recovery_triangles[constrained_triangles]
    ]
    seam_fragments: tuple[PeriodicPolyhedralSeamFragment, ...] = ()
    assembly = None
    if periodic_group is not None:
        from ..discretization._coordinate_enclosure import coordinate_enclosure_budget

        old_bounds = dict(zip(map(int, order), diameters, strict=True))
        remaining = policy.maximum_work_units - spent_work
        seam_ledger = coordinate_enclosure_budget(remaining, policy.maximum_scratch_bytes)
        before = seam_ledger.work_units
        with (
            seam_ledger.activate(),
            seam_ledger.bound_stage(remaining, policy.maximum_scratch_bytes),
        ):
            seam_fragments = prepare_periodic_polyhedral_seam_fragments(
                mesh,
                geometry,
                face_facets,
                periodic_values,
                maximum_work=remaining,
                maximum_scratch_bytes=policy.maximum_scratch_bytes,
            )
            assembly = _reconcile_periodic_polyhedral(
                mesh,
                geometry,
                diagram,
                seam_fragments,
                periodic_values,
                periodic_group,
                policy.maximum_vertices,
                seam_ledger,
            )
            validity = certify_cell_geometry_validity(
                assembly.geometry, mesh=assembly.mesh
            )
            bad = np.flatnonzero(
                np.asarray(validity.status) != int(CellValidityStatus.CERTIFIED_VALID)
            )
            if bad.size:
                raise PolyhedralGenerationError(
                    "periodic_piece_geometry_unresolved", entities=tuple(map(int, bad))
                )
            mesh, geometry = assembly.mesh, assembly.geometry
            face_facets = face_facets[assembly.parent_faces]
            face_recovery_triangles = face_recovery_triangles[assembly.parent_faces]
            order = np.asarray(
                _polyhedral_connectivity(mesh).cell_global_ids, dtype=np.int64
            )
            regions = data.cell_regions[order]
            diameters = np.asarray([old_bounds[int(cell)] for cell in order])
            exact_volumes, _, _ = _certified_cell_measures(
                mesh, geometry, maximum_work=remaining
            )
            achieved = np.bincount(
                diagram.cell_original_sites[order],
                weights=exact_volumes,
                minlength=site_array.shape[0],
            )
            residual = (
                float(np.max(np.abs(achieved - targets) / targets))
                if targets.size
                else 0.0
            )
            if targets.size and residual > policy.relative_volume_tolerance:
                raise PolyhedralGenerationError(
                    "periodic_volume_optimization_unresolved",
                    achieved={"relative_volume_residual": residual},
                )
            seam_ledger.charge_native_work(
                seam_ledger.work_units - seam_ledger.native_charged_work_units
            )
        spent_work += seam_ledger.work_units - before
    pairs = np.full((face_facets.size, 2), -1, dtype=np.int64)
    constrained = face_facets >= 0
    pairs[constrained] = complex_.facet_regions[face_facets[constrained]]
    interfaces = np.flatnonzero(
        constrained
        & (pairs[:, 0] >= 0)
        & (pairs[:, 1] >= 0)
        & (pairs[:, 0] != pairs[:, 1])
    )
    sheets = np.flatnonzero(constrained & (pairs[:, 0] == pairs[:, 1]))
    feature_source_edges = _source_feature_edges(complex_, recovery)
    feature_sources, feature_distances, feature_gaps, feature_work = _feature_ancestry(
        mesh,
        geometry,
        recovery,
        diagram,
        feature_source_edges,
        policy.maximum_work_units - spent_work,
        vertex_carriers=None if assembly is None else assembly.vertex_carriers,
    )
    spent_work += feature_work
    combined = feature_distances + feature_gaps
    combined = np.where(combined == 0, 0.0, np.nextafter(combined, np.inf))
    bad_features = np.flatnonzero(combined > policy.feature_tolerance)
    if bad_features.size:
        raise PolyhedralGenerationError(
            "feature_coverage_unresolved",
            entities=tuple(map(int, bad_features)),
            achieved=feature_gaps[bad_features],
        )
    # Consumer-admissibility splitting is not a new connected site component.
    # Rejoin its lineage through reciprocal unconstrained same-site faces.
    parent = np.arange(data.cell_sites.size, dtype=np.int64)

    def root(cell: int) -> int:
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return int(cell)

    for face in range(data.face_cells.shape[0]):
        left, right = int(data.face_cells[face, 0]), int(data.face_cells[face, 1])
        if (
            right >= 0
            and data.face_facets[face] < 0
            and diagram.cell_original_sites[left] == diagram.cell_original_sites[right]
        ):
            a, b = root(left), root(right)
            parent[max(a, b)] = min(a, b)
    if assembly is not None:
        connectivity = _polyhedral_connectivity(mesh)
        owners = np.asarray(connectivity.face_owner)
        for source_face, target_face, _ in assembly.face_pairs:
            left, right = int(order[owners[source_face]]), int(order[owners[target_face]])
            if diagram.cell_original_sites[left] == diagram.cell_original_sites[right]:
                a, b = root(left), root(right)
                parent[max(a, b)] = min(a, b)
    roots = np.asarray([root(cell) for cell in range(parent.size)])
    components = np.zeros_like(diagram.cell_original_sites)
    for site in np.unique(diagram.cell_original_sites):
        selected = np.flatnonzero(diagram.cell_original_sites == site)
        _, inverse = np.unique(roots[selected], return_inverse=True)
        components[selected] = inverse
    identity = canonical_fingerprint(
        {
            "kind": "native-polyhedral-construction",
            "source": domain.domain_id,
            "mesh": mesh.mesh_id,
            "geometry": cell_geometry_id(geometry),
            "sites": array_tree_fingerprint((site_array, values)),
            "pieces": array_tree_fingerprint(
                (data.piece_sites, data.piece_tets, data.piece_cells, data.piece_volumes)
            ),
        }
    )
    return PolyhedralConstruction(
        mesh,
        geometry,
        domain,
        recovery,
        diagram,
        _freeze(regions),
        _freeze(diagram.cell_original_sites[order]),
        _freeze(face_facets),
        _freeze(face_recovery_triangles),
        _freeze(interfaces),
        _freeze(sheets),
        _freeze(feature_sources),
        _freeze(feature_source_edges),
        _freeze(feature_distances),
        _freeze(feature_gaps),
        _freeze(parents),
        _freeze(components[order]),
        _freeze(split),
        _freeze(targets),
        _freeze(achieved),
        refinement_steps,
        lloyd_steps,
        residual,
        float(np.max(diameters)),
        identity,
        spent_work,
        periodic_constraints=periodic_values,
        seam_fragments=seam_fragments,
        seam_face_pairs=None if assembly is None else _freeze(assembly.face_pairs),
        seam_vertex_pairs=None if assembly is None else _freeze(assembly.vertex_pairs),
    )


__all__ = [
    "NativePolyhedralSchedule",
    "PolyhedralConstruction",
    "PolyhedralGenerationError",
    "PeriodicPolyhedralSeamFragment",
    "prepare_periodic_polyhedral_seam_fragments",
    "generate_polyhedral_volume",
]
