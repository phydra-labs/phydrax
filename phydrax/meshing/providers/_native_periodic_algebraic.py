#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from itertools import product
from math import ceil, floor, gcd, isqrt, lcm
from typing import TYPE_CHECKING

import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...discretization._cell_mesh import CellMesh
from ...discretization._exact_algebraic_geometry import ExactScalar, RadicalElement
from ...discretization._periodic_cell import PeriodicCell
from ...discretization._periodic_exact import exact_periodic_element
from ...discretization._periodic_topology import _entity_vertex_rows, PeriodicMeshTopology
from ...typing import checked
from .._contracts import SurfaceMeshingSpec, VolumeMeshingSpec
from .._controls import FeatureKind
from .._periodic import PeriodicPointOrbits
from .._sizing import SizeCompliancePolicy, SizeControlStrength, UniformSizeControl

if TYPE_CHECKING:
    from ._native_periodic import NativePeriodicSource


def _exact_point(
    matrix: tuple[tuple[Fraction, ...], ...], point: np.ndarray, /
) -> tuple[Fraction, ...]:
    values = tuple(Fraction.from_float(float(item)) for item in point) + (Fraction(1),)
    dimension = point.size
    return tuple(
        sum(
            (matrix[row][column] * values[column] for column in range(dimension + 1)),
            Fraction(),
        )
        for row in range(dimension)
    )


@checked
def periodic_quotient_edge_squared_lengths(
    mesh: CellMesh,
    /,
) -> tuple[np.ndarray, np.ndarray, tuple[ExactScalar, ...]]:
    """Exact squared lengths of each quotient edge, preserving its winding action."""
    topology = mesh.periodic_topology
    if not isinstance(topology, PeriodicMeshTopology):
        raise ValueError("mesh must retain PeriodicMeshTopology.")
    topology.validate_restored()
    topology.require_lifted(mesh.topology, mesh.topology_id, mesh.blocks)
    topology.require_coordinates(mesh.coordinates)
    edge_rows = _entity_vertex_rows(mesh)[1]
    lifted_offsets = np.asarray(topology.lifted_offsets, dtype=np.int64)
    quotient_offsets = np.asarray(topology.quotient_offsets, dtype=np.int64)
    quotient_begin, quotient_end = map(int, quotient_offsets[1:3])
    representatives = np.asarray(topology.representatives, dtype=np.int64)[
        quotient_begin:quotient_end
    ]
    if representatives.size != quotient_end - quotient_begin:
        raise ValueError("Periodic quotient edge representative table is malformed.")
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    exact_source = topology.exact_coordinate_source
    exact_points = None if exact_source is None else exact_source.source_coordinates()
    vertex_representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
    vertex_shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    vertex_orbits = (
        np.asarray(topology.orbit_indices, dtype=np.int64)[lifted_offsets[0]:lifted_offsets[1]]
        - quotient_offsets[0]
    )
    vertex_ids = np.asarray(topology.quotient.entity_sets[0].entity_ids, dtype=np.int64)
    edge_ids = np.asarray(topology.quotient.entity_sets[1].entity_ids, dtype=np.int64)
    endpoints = np.empty((representatives.size, 2), dtype=np.int64)
    squared: list[ExactScalar] = []
    for row, lifted_edge in enumerate(representatives):
        if not 0 <= lifted_edge < len(edge_rows):
            raise ValueError("Periodic quotient edge representative is out of range.")
        vertices = np.asarray(edge_rows[int(lifted_edge)], dtype=np.int64)
        if vertices.shape != (2,):
            raise ValueError("Periodic quotient edges must have two vertices.")
        first, second = map(int, vertices)
        endpoints[row] = (
            vertex_ids[vertex_orbits[first]],
            vertex_ids[vertex_orbits[second]],
        )
        if exact_points is None:
            first_action = exact_periodic_element(topology.cell, vertex_shifts[first])
            second_action = exact_periodic_element(topology.cell, vertex_shifts[second])
            first_point = _exact_point(first_action, points[vertex_representatives[first]])
            second_point = _exact_point(second_action, points[vertex_representatives[second]])
        else:
            first_point, second_point = exact_points[first], exact_points[second]
        squared.append(
            sum(
                ((second_point[axis] - first_point[axis]) ** 2 for axis in range(points.shape[1])),
                Fraction(),
            )
        )
    if edge_ids.shape != (len(squared),):
        raise ValueError("Periodic quotient edge IDs do not align with representative rows.")
    return edge_ids, endpoints, tuple(squared)



type Point = tuple[RadicalElement, RadicalElement]
type Image = tuple[int, int]
type Edge = tuple[int, int, int, int]
type Corner = tuple[int, Image]
type SiteIdentity = tuple[str, tuple[int, ...]]
type _Vector = tuple[Fraction, Fraction]

@dataclass(frozen=True, slots=True)
class ExactAlgebraicCarrier:
    exact_coordinates: tuple[Point, ...]
    points: np.ndarray
    simplices: np.ndarray
    shifts: np.ndarray
    regions: np.ndarray
    ancestry: tuple[tuple[int, ...], ...]
    site_identities: tuple[SiteIdentity, ...]
    source_identity: str

@dataclass(frozen=True, slots=True)
class _Strata:
    points: tuple[Point, ...]
    cells: np.ndarray
    shifts: np.ndarray
    regions: np.ndarray
    edges: np.ndarray
    cell: PeriodicCell
    root_ids: np.ndarray

@dataclass(frozen=True, slots=True)
class _Grid:
    strata: _Strata
    points: tuple[Point, ...]
    vectors: tuple[Point, Point]
    dual: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
    rows: tuple[tuple[int, ...], ...]
    vertical: tuple[Edge, ...]
    incidence: dict[Edge, tuple[tuple[int, int], ...]]
    protected: tuple[Edge, ...]
    omitted_component: frozenset[int]
    omitted_region: int
    radius: Fraction
    maximum: Fraction
    run: RadicalElement
    height: RadicalElement

class _Predicates:
    def __init__(self) -> None:
        self.boxes: dict[tuple[Point, Point], tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]] = {}

    def box(self, a: Point, b: Point) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
        key = (a, b)
        if key not in self.boxes:
            ax, ay = a[0].interval(bits=32), a[1].interval(bits=32)
            bx, by = b[0].interval(bits=32), b[1].interval(bits=32)
            self.boxes[key] = (
                (min(ax[0], bx[0]), max(ax[1], bx[1])),
                (min(ay[0], by[0]), max(ay[1], by[1])),
            )
        return self.boxes[key]

    def orient(self, a: Point, b: Point, c: Point) -> int:
        return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).sign()


def _image(row: np.ndarray, /) -> Image:
    return (int(row[0]), int(row[1]))


def _radical_point(row: Sequence[ExactScalar], /) -> Point:
    if len(row) != 2:
        raise ValueError("Source-action construction requires planar source coordinates.")
    x, y = row
    return (
        x if isinstance(x, RadicalElement) else RadicalElement(x),
        y if isinstance(y, RadicalElement) else RadicalElement(y),
    )


def prepare_periodic_source_strata(mesh: CellMesh, regions: np.ndarray | None, features: tuple, /) -> _Strata:
    """Extract quotient cells and protected lifted-edge orbits using canonical IDs."""
    topology = mesh.periodic_topology
    if topology is None or not isinstance(topology.cell, PeriodicCell) or regions is None:
        raise ValueError("Material source requires a translational quotient and authored regions.")
    topology.require_lifted(mesh.topology, mesh.topology_id, mesh.blocks)
    topology.require_coordinates(mesh.coordinates)
    offsets = np.asarray(topology.lifted_offsets, dtype=np.int64)
    qoffsets = np.asarray(topology.quotient_offsets, dtype=np.int64)
    orbits = np.asarray(topology.orbit_indices, dtype=np.int64)
    representatives = np.asarray(topology.representatives, dtype=np.int64)
    vertex_rows = orbits[offsets[0]:offsets[1]] - qoffsets[0]
    quotient_ids = np.asarray(topology.quotient.entity_sets[0].entity_ids, dtype=np.int64)
    order = np.argsort(quotient_ids, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    quotient_representatives = representatives[qoffsets[0]:qoffsets[1]][order]
    roots = np.asarray(topology.vertex_representatives, dtype=np.int64)[quotient_representatives]
    root_ids = quotient_ids[order]
    vertex_roots = inverse[vertex_rows]
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    if shifts.ndim != 2 or shifts.shape[1] != 2:
        raise ValueError("Source-action construction requires planar deck shifts.")
    if np.any(shifts[roots]):
        raise ValueError("Canonical source representatives must have zero deck shift.")
    rows = _entity_vertex_rows(mesh)
    leaders = representatives[qoffsets[2]:qoffsets[3]]
    lifted_cell_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    cells_by_id = {
        int(identifier): np.asarray(row, dtype=np.int64)
        for block in mesh.blocks
        for identifier, row in zip(block.global_ids, block.vertices, strict=True)
    }
    cell_rows = [cells_by_id[int(lifted_cell_ids[i])] for i in leaders]
    if any(row.shape != (3,) for row in cell_rows):
        raise ValueError("Source-action construction requires simplex cells.")
    block_ids = np.concatenate([np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks])
    values = np.asarray(regions, dtype=np.int64)
    if values.shape != block_ids.shape:
        raise ValueError("Source material rows do not match cells.")
    region_by_id = dict(zip(map(int, block_ids), map(int, values), strict=True))
    selected_regions = np.asarray([region_by_id[int(lifted_cell_ids[i])] for i in leaders], dtype=np.int64)
    cell_orbits = orbits[offsets[2]:offsets[3]] - qoffsets[2]
    for row, orbit in enumerate(cell_orbits):
        if region_by_id[int(lifted_cell_ids[row])] != selected_regions[int(orbit)]:
            raise ValueError("Periodic copies of a source cell must have the same material.")
    cells = np.asarray([vertex_roots[row] for row in cell_rows], dtype=np.int64)
    corner_shifts = np.asarray([shifts[row] for row in cell_rows], dtype=np.int64)
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    selected_ids = set()
    for feature in features:
        selected_ids.update(map(int, np.asarray(feature.scope.entity_ids, dtype=np.int64)))
    if not selected_ids <= set(map(int, edge_ids)):
        raise ValueError("Protected source scope contains unknown lifted edges.")
    protected = set()
    for identifier, row in zip(edge_ids, rows[1], strict=True):
        if int(identifier) in selected_ids:
            first, second = map(int, row)
            protected.add(_key(int(vertex_roots[first]), _image(shifts[first]), int(vertex_roots[second]), _image(shifts[second]))[0])
    exact_source = topology.exact_coordinate_source
    if exact_source is None:
        source_points = tuple(
            _radical_point(tuple(Fraction(float(value)) for value in row))
            for row in np.asarray(mesh.coordinates, dtype=np.float64)[roots]
        )
    else:
        bank = exact_source.source_coordinates()
        source_points = tuple(_radical_point(bank[int(root)]) for root in roots)
    return _Strata(source_points, cells, corner_shifts, selected_regions, np.asarray(sorted(protected), dtype=np.int64).reshape((-1, 4)), topology.cell, root_ids)


def _compatible_edges(
    edges: list[Edge],
    segments: list[_Segment],
    penalties: list[tuple[int, int, RadicalElement]],
    grid: _Grid,
    work: _Predicates,
    *,
    below_limit: int | None = None,
    above_limit: int | None = None,
) -> tuple[Edge, ...]:
    return _compatible_edges_search(edges, segments, penalties, grid, work, below_limit=below_limit, above_limit=above_limit)


def _key(a: int, ia: Image, b: int, ib: Image) -> tuple[Edge, int]:
    if (a, *ia) <= (b, *ib):
        return ((a, b, ib[0] - ia[0], ib[1] - ia[1]), 1)
    return ((b, a, ia[0] - ib[0], ia[1] - ib[1]), -1)

def _add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])

def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])

def _scale(a: Point, factor: RadicalElement | Fraction | int) -> Point:
    return (a[0] * factor, a[1] * factor)

def _square(a: Point, b: Point) -> RadicalElement:
    x, y = _sub(b, a)
    return x * x + y * y

def _lift(point: Point, image: Image, vectors: tuple[Point, Point]) -> Point:
    return _add(point, _add(_scale(vectors[0], image[0]), _scale(vectors[1], image[1])))

def _fractional(point: Point, grid: _Grid | OrbitLatticeBands) -> Point:
    first, second = grid.dual
    return (point[0] * first[0] + point[1] * first[1], point[0] * second[0] + point[1] * second[1])

def _wrap(point: Point, grid: _Grid) -> Point:
    fractional = _fractional(point, grid)
    return _lift(point, (-floor(fractional[0]), -floor(fractional[1])), grid.vectors)

def _recover_grid(source: NativePeriodicSource, specification: SurfaceMeshingSpec | VolumeMeshingSpec, work: _Predicates) -> _Grid | None:
    """Recover one source-action configuration, or refuse the construction's premises.

    The admission proof intentionally keeps its commuting actions, metric,
    material cycles, and circle-profile branch together: every later check
    depends on the same recovered source-incidence configuration and charge order.
    """
    match source.domain:
        case CellMesh() as mesh:
            pass
        case _:
            return None
    if source.ambient_dimension != 2 or mesh.topological_dimension != 2:
        return None
    topology = mesh.periodic_topology
    if topology is None:
        return None
    match topology.cell:
        case PeriodicCell():
            pass
        case _:
            return None
    policy = specification.size_compliance
    if policy.absolute_tolerance != 0 or policy.relative_tolerance != 0 or set(policy.target_statistics) != {'p50', 'p95'}:
        return None
    controls: list[UniformSizeControl] = []
    for control in specification.size_controls:
        match control:
            case UniformSizeControl() if control.strength is SizeControlStrength.HARD:
                controls.append(control)
            case _:
                return None
    if not controls or not specification.protected_features:
        return None
    if any(
        not feature.hard
        or feature.feature_kind is not FeatureKind.MATERIAL_INTERFACE
        or feature.scope.entity_dimension != 1
        or feature.scope.source_id != source.source_id
        or feature.scope.source_revision != source.source_revision
        or feature.scope.entity_set_id != mesh.entity_set(1).entity_set_id
        for feature in specification.protected_features
    ):
        return None
    if source.cell_regions is None:
        return None
    count = np.unique(np.asarray(topology.vertex_representatives)).size
    n = isqrt(count)
    if n < 4 or n % 2 or n * n != count:
        return None
    strata = prepare_periodic_source_strata(mesh, source.cell_regions, specification.protected_features)
    if any(
        len(value.terms) > 1 or (value.terms and value.terms[0][0] != 1)
        for point in strata.points for value in point
    ):
        return None
    raw_vectors = np.asarray(strata.cell.vectors)
    rational_vectors = ((Fraction(float(raw_vectors[0, 0])), Fraction(float(raw_vectors[0, 1]))), (Fraction(float(raw_vectors[1, 0])), Fraction(float(raw_vectors[1, 1]))))
    a, b = rational_vectors
    determinant = a[0] * b[1] - a[1] * b[0]
    norm = a[0] * a[0] + a[1] * a[1]
    if determinant <= 0:
        return None
    radius = Fraction(float(controls[0].target_size))
    if radius * radius != norm / (n * n):
        return None
    material_ids = set((int(value) for value in strata.regions))
    if len(material_ids) != 2 or any(
        Fraction(float(control.target_size)) != radius
        or control.minimum_size is not None
        or control.maximum_growth_rate is not None
        or control.scope.entity_dimension != 0
        or control.scope.source_id != source.source_id
        or control.scope.source_revision != source.source_revision
        or control.scope.entity_set_id != topology.quotient.entity_sets[0].entity_set_id
        or set(map(int, control.scope.entity_ids)) != set(map(int, strata.root_ids))
        for control in controls
    ):
        return None
    if any((control.maximum_size is None for control in controls)):
        return None
    maximum = min((Fraction(float(control.maximum_size)) for control in controls if control.maximum_size is not None))
    if maximum < radius or maximum > 3 * radius / 2:
        return None
    dual = ((b[1] / determinant, -b[0] / determinant), (-a[1] / determinant, a[0] / determinant))
    if any((maximum * (abs(row[0]) + abs(row[1])) >= Fraction(1, 2) for row in dual)):
        return None
    points = strata.points
    vectors = ((RadicalElement(a[0]), RadicalElement(a[1])), (RadicalElement(b[0]), RadicalElement(b[1])))
    incidence_lists: dict[Edge, list[tuple[int, int]]] = {}
    actions: list[dict[int, tuple[int, Image]]] = [{}, {}]
    for ci, (cell, images) in enumerate(zip(strata.cells, strata.shifts, strict=True)):
        corners = [_lift(points[int(root)], (int(image[0]), int(image[1])), vectors) for root, image in zip(cell, images, strict=True)]
        if work.orient(*corners) <= 0:
            return None
        for first, second in ((0, 1), (1, 2), (2, 0)):
            ia = (int(images[first, 0]), int(images[first, 1]))
            ib = (int(images[second, 0]), int(images[second, 1]))
            edge, sign = _key(int(cell[first]), ia, int(cell[second]), ib)
            incidence_lists.setdefault(edge, []).append((ci, sign))
    if any((len(members) != 2 or sum((sign for _, sign in members)) != 0 for members in incidence_lists.values())):
        return None
    for edge in sorted(incidence_lists):
        first, second, sx, sy = edge
        delta = _sub(_lift(points[second], (sx, sy), vectors), points[first])
        for axis in range(2):
            for direction in (-1, 1):
                if _scale(delta, direction * n) == vectors[axis]:
                    root, target = (first, second) if direction == 1 else (second, first)
                    image = (direction * sx, direction * sy)
                    old = actions[axis].get(root)
                    if old is not None and old != (target, image):
                        return None
                    actions[axis][root] = (target, image)
    if any((len(action) != count for action in actions)):
        return None
    anchor = min(actions[0])
    rows: list[tuple[int, ...]] = []
    vertical: list[Edge] = []
    root = anchor
    column_image = (0, 0)
    seen: set[int] = set()
    for column in range(n):
        row_roots: list[int] = []
        current, winding = (root, (0, 0))
        for row in range(n):
            if current in seen:
                return None
            seen.add(current)
            row_roots.append(current)
            target, shift = actions[1][current]
            vertical.append(_key(current, (0, 0), target, shift)[0])
            xroot, ximage = actions[0][current]
            yxroot, yximage = actions[1][xroot]
            xyroot, xyimage = actions[0][target]
            if yxroot != xyroot or tuple((ximage[k] + yximage[k] for k in range(2))) != tuple((shift[k] + xyimage[k] for k in range(2))):
                return None
            current = target
            winding = (winding[0] + shift[0], winding[1] + shift[1])
        if current != root or winding != (0, 1):
            return None
        rows.append(tuple(row_roots))
        root, shift = actions[0][root]
        column_image = (column_image[0] + shift[0], column_image[1] + shift[1])
    if root != anchor or column_image != (1, 0) or len(seen) != count:
        return None
    protected = tuple(((int(row[0]), int(row[1]), int(row[2]), int(row[3])) for row in strata.edges))
    material_edges = {edge for edge, members in incidence_lists.items() if len({int(strata.regions[ci]) for ci, _ in members}) == 2}
    if set(protected) != material_edges or not material_edges <= set(vertical) or len(protected) != 2 * n:
        return None
    components = [frozenset(row) for column, row in enumerate(rows) if all((edge in material_edges for edge in vertical[column * n:(column + 1) * n]))]
    if len(components) != 2:
        return None
    component_columns = [i for i, row in enumerate(rows) if frozenset(row) in components]
    if (component_columns[1] - component_columns[0]) % n != n // 2:
        return None
    shear = (a[0] * b[0] + a[1] * b[1]) / norm
    aspect = determinant / norm
    if not 0 < shear < Fraction(1, 2) or shear * shear + aspect * aspect >= 4:
        return None
    h, height_total = (Fraction(1, n), aspect / n)
    closure = shear / n + 2 * h
    length2 = closure * closure + height_total * height_total
    intercept = (8 * h * h + length2) / 6
    radicand = h * h * length2 - intercept * intercept
    if closure <= 0 or height_total <= 0 or radicand <= 0:
        return None
    radical = RadicalElement.sqrt(radicand)
    run = (intercept * closure + height_total * radical) / length2
    height = (intercept * height_total - closure * radical) / length2
    if not (0 < height < height_total / 4 and run > 0 and (3 * run - closure > 0) and (height_total - 3 * height > 0)):
        return None
    return _Grid(strata, points, vectors, dual, tuple(rows), tuple(vertical), {edge: tuple(members) for edge, members in incidence_lists.items()}, protected, max(components, key=min), min(material_ids), radius, maximum, run, height)

def _sites(grid: _Grid, work: _Predicates) -> tuple[tuple[Point, ...], tuple[SiteIdentity, ...]]:
    points = list(grid.points)
    identities: list[SiteIdentity] = [('authored_source_root', (int(grid.strata.root_ids[root]),)) for root in range(len(points))]
    normal = (-grid.vectors[0][1], grid.vectors[0][0])
    step = _add(_scale(grid.vectors[0], grid.run), _scale(normal, grid.height))
    n = len(grid.rows)
    for row in range(n):
        for layer in range(1, 4):
            for column in range(n):
                root = grid.rows[column][row]
                edge = grid.vertical[column * n + row]
                points.append(_wrap(_add(grid.points[root], _scale(step, layer)), grid))
                identities.append(('source_edge_band', (int(grid.strata.root_ids[edge[0]]), int(grid.strata.root_ids[edge[1]]), *edge[2:], int(grid.strata.root_ids[root]), layer)))
    for edge in grid.protected:
        first, second, sx, sy = edge
        a, b = (grid.points[first], _lift(grid.points[second], (sx, sy), grid.vectors))
        delta = _sub(b, a)
        square = _square(a, b)
        multiplier2 = grid.radius * grid.radius / square - Fraction(1, 4)
        if multiplier2 <= 0:
            raise ValueError('periodic.algebraic.protected_apex must have two circle intersections; got an invalid source edge')
        if len(multiplier2.terms) != 1 or multiplier2.terms[0][0] != 1:
            raise ValueError('periodic.algebraic.source_edge must be rational; got a radical source edge')
        multiplier = RadicalElement.sqrt(multiplier2.terms[0][1])
        for side in (-1, 1):
            side_region = next((int(grid.strata.regions[ci]) for ci, sign in grid.incidence[edge] if sign == side))
            if first in grid.omitted_component and side_region == grid.omitted_region:
                continue
            point = _add(_scale(_add(a, b), Fraction(1, 2)), _scale((-delta[1], delta[0]), side * multiplier))
            points.append(_wrap(point, grid))
            identities.append(('protected_edge_side_material', (int(grid.strata.root_ids[edge[0]]), int(grid.strata.root_ids[edge[1]]), *edge[2:], side, side_region)))
    if len(set(points)) != len(points):
        raise ValueError('periodic.algebraic.sites must have distinct field-valued orbit coordinates; got coincident constructed sites')
    return (tuple(points), tuple(identities))

@dataclass(frozen=True, slots=True)
class _Segment:
    a: Point
    b: Point
    fa: Point
    fb: Point

def _segment(edge: Edge, points: tuple[Point, ...], grid: _Grid) -> _Segment:
    a = points[edge[0]]
    b = _lift(points[edge[1]], (edge[2], edge[3]), grid.vectors)
    return _Segment(a, b, _fractional(a, grid), _fractional(b, grid))

class _PeriodicBins:
    """Index closed rational enclosures on the unit torus without image banks."""

    def __init__(self, segments: list[_Segment], work: _Predicates) -> None:
        self.work = work
        self.segments = segments
        self.width = max(1, isqrt(len(segments)))
        self.bins: dict[tuple[int, int], int] = {}
        for index, segment in enumerate(segments):
            bit = 1 << index
            for key in self._keys(segment):
                self.bins[key] = self.bins.get(key, 0) | bit

    def _keys(self, segment: _Segment) -> Iterator[tuple[int, int]]:
        box = self.work.box(segment.fa, segment.fb)
        first, second = (range(floor(lo * self.width), min(floor(hi * self.width) + 1, floor(lo * self.width) + self.width)) for lo, hi in box)
        for x in first:
            for y in second:
                yield (x % self.width, y % self.width)

    def candidates(self, segment: _Segment, *, stop: int | None=None) -> Iterator[int]:
        mask = 0
        for key in self._keys(segment):
            mask |= self.bins.get(key, 0)
        if stop is not None:
            mask &= (1 << stop) - 1
        box = self.work.box(segment.fa, segment.fb)
        while mask:
            bit = mask & -mask
            index = bit.bit_length() - 1
            mask ^= bit
            other = self.segments[index]
            other_box = self.work.box(other.fa, other.fb)
            if all((ceil(box[k][0] - other_box[k][1]) <= floor(box[k][1] - other_box[k][0]) for k in range(2))):
                yield index

def _conflict_graph(segments: list[_Segment], grid: _Grid, work: _Predicates) -> list[int]:
    count = len(segments)
    adjacency = [0] * count
    index = _PeriodicBins(segments, work)
    for i, segment in enumerate(segments):
        for j in index.candidates(segment, stop=i):
            if _conflict(segment, segments[j], grid, work):
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    return adjacency

def _images(first: _Segment, second: _Segment, work: _Predicates) -> Iterator[Image]:
    first_box = work.box(first.fa, first.fb)
    second_box = work.box(second.fa, second.fb)
    bounds = tuple(((ceil(first_box[k][0] - second_box[k][1]), floor(first_box[k][1] - second_box[k][0])) for k in range(2)))
    yield from product(range(bounds[0][0], bounds[0][1] + 1), range(bounds[1][0], bounds[1][1] + 1))

def _interior(a: Point, b: Point, p: Point, work: _Predicates) -> bool:
    return p != a and p != b and all((min(a[k], b[k]) <= p[k] <= max(a[k], b[k]) for k in range(2))) and (work.orient(a, b, p) == 0)

def _conflict(first: _Segment, second: _Segment, grid: _Grid, work: _Predicates) -> bool:
    for image in _images(first, second, work):
        c, d = (_lift(second.a, image, grid.vectors), _lift(second.b, image, grid.vectors))
        a, b = (first.a, first.b)
        if any((max(a[k], b[k]) < min(c[k], d[k]) or max(c[k], d[k]) < min(a[k], b[k]) for k in range(2))):
            continue
        ab_c, ab_d = (work.orient(a, b, c), work.orient(a, b, d))
        cd_a, cd_b = (work.orient(c, d, a), work.orient(c, d, b))
        if ab_c * ab_d < 0 and cd_a * cd_b < 0:
            return True
        if ab_c == ab_d == cd_a == cd_b == 0:
            axis = 0 if a[0] != b[0] else 1
            if max(min(a[axis], b[axis]), min(c[axis], d[axis])) < min(max(a[axis], b[axis]), max(c[axis], d[axis])):
                return True
        if _interior(a, b, c, work) or _interior(a, b, d, work) or _interior(c, d, a, work) or _interior(c, d, b, work):
            return True
    return False

def _compatible_edges_search(edges: list[Edge], segments: list[_Segment], penalties: list[tuple[int, int, RadicalElement]], grid: _Grid, work: _Predicates, *, below_limit: int | None, above_limit: int | None) -> tuple[Edge, ...]:
    short_limit = len(edges) if below_limit is None else below_limit
    long_limit = len(edges) if above_limit is None else above_limit
    adjacency = _conflict_graph(segments, grid, work)
    cache: dict[int, tuple[int, dict[tuple[int, int], tuple[RadicalElement, int]]]] = {0: (0, {(0, 0): (RadicalElement(), 0)})}

    def retain(frontier: dict[tuple[int, int], tuple[RadicalElement, int]], tails: tuple[int, int], cost: RadicalElement, selected: int) -> None:
        if tails[0] > short_limit or tails[1] > long_limit:
            return
        old = frontier.get(tails)
        if old is None:
            frontier[tails] = (cost, selected)
        elif cost < old[0]:
            frontier[tails] = (cost, selected)

    def independent(mask: int) -> tuple[int, dict[tuple[int, int], tuple[RadicalElement, int]]]:
        old = cache.get(mask)
        if old is not None:
            return old
        isolates, best, degree = (0, -1, -1)
        scan = mask
        while scan:
            bit = scan & -scan
            vertex = bit.bit_length() - 1
            current = (adjacency[vertex] & mask).bit_count()
            if current == 0:
                isolates |= bit
            elif current > degree:
                best, degree = (vertex, current)
            scan ^= bit
        frontier: dict[tuple[int, int], tuple[RadicalElement, int]] = {}
        if isolates:
            size, child = independent(mask ^ isolates)
            short_count, long_count, square = (0, 0, RadicalElement())
            scan = isolates
            while scan:
                bit = scan & -scan
                below, above, length = penalties[bit.bit_length() - 1]
                short_count += below
                long_count += above
                square += length
                scan ^= bit
            for tails, (cost, selected) in child.items():
                retain(frontier, (tails[0] + short_count, tails[1] + long_count), cost + square, selected | isolates)
            result = (size + isolates.bit_count(), frontier)
        else:
            component = pending = mask & -mask
            while pending:
                bit = pending & -pending
                pending ^= bit
                neighbors = adjacency[bit.bit_length() - 1] & mask & ~component
                component |= neighbors
                pending |= neighbors
            if component != mask:
                left_size, left = independent(component)
                right_size, right = independent(mask ^ component)
                for lt, (lc, lm) in left.items():
                    for rt, (rc, rm) in right.items():
                        retain(frontier, (lt[0] + rt[0], lt[1] + rt[1]), lc + rc, lm | rm)
                result = (left_size + right_size, frontier)
            else:
                bit = 1 << best
                included_size, included = independent(mask & ~bit & ~adjacency[best])
                excluded_size, excluded = independent(mask & ~bit)
                size = max(included_size + 1, excluded_size)
                if included_size + 1 == size:
                    below, above, length = penalties[best]
                    for tails, (cost, selected) in included.items():
                        retain(frontier, (tails[0] + below, tails[1] + above), cost + length, selected | bit)
                if excluded_size == size:
                    for tails, (cost, selected) in excluded.items():
                        retain(frontier, tails, cost, selected)
                result = (size, frontier)
        cache[mask] = result
        return result
    try:
        _, frontier = independent((1 << len(edges)) - 1)
        if not frontier:
            raise ValueError(f'periodic.algebraic.completion must satisfy requested quantile tail bounds; got no maximum-cardinality completion with short_limit={short_limit!r}, long_limit={long_limit!r}')
        _, selected = min(frontier.values())
        return tuple((edge for i, edge in enumerate(edges) if selected & 1 << i))
    finally:
        cache.clear()
        adjacency.clear()
        del independent

def _target_graph_search(points: tuple[Point, ...], grid: _Grid, work: _Predicates, quantiles: tuple[Fraction, ...]) -> tuple[Edge, ...]:
    protected = {edge: _segment(edge, points, grid) for edge in grid.protected}
    fractional = tuple((_fractional(point, grid) for point in points))
    bounds = tuple((grid.maximum * (abs(row[0]) + abs(row[1])) for row in grid.dual))
    singletons = [_Segment(point, point, fp, fp) for point, fp in zip(points, fractional, strict=True)]
    singleton_index = _PeriodicBins(singletons, work)
    protected_edges = list(protected)
    protected_segments = list(protected.values())
    protected_index = _PeriodicBins(protected_segments, work)
    radius2 = grid.radius * grid.radius
    maximum2 = grid.maximum * grid.maximum
    bank: list[tuple[Edge, _Segment, RadicalElement]] = []
    for a in range(len(points)):
        for b in range(a + 1, len(points)):
            ranges = tuple((range(ceil(fractional[a][k] - fractional[b][k] - bounds[k]), floor(fractional[a][k] - fractional[b][k] + bounds[k]) + 1) for k in range(2)))
            for sx, sy in product(ranges[0], ranges[1]):
                edge = (a, b, sx, sy)
                stop = _lift(points[b], (sx, sy), grid.vectors)
                square = _square(points[a], stop)
                if square <= maximum2:
                    segment = _segment(edge, points, grid)
                    bank.append((edge, segment, square))

    def visible(edge: Edge, segment: _Segment) -> bool:
        for root in singleton_index.candidates(segment):
            singleton = singletons[root]
            if root in edge[:2]:
                continue
            if any((_interior(segment.a, segment.b, _lift(singleton.a, image, grid.vectors), work) for image in _images(segment, singleton, work))):
                return False
        return True
    radius_edges: list[Edge] = []
    radius_segments: list[_Segment] = []
    for edge, segment, square in bank:
        if square != radius2:
            continue
        if any((_conflict(segment, protected_segments[index], grid, work) for index in protected_index.candidates(segment) if protected_edges[index] != edge)) or not visible(edge, segment):
            continue
        radius_edges.append(edge)
        radius_segments.append(segment)
    selected_radius = _compatible_edges(radius_edges, radius_segments, [(0, 0, RadicalElement()) for _ in radius_edges], grid, work)
    mandatory = dict(protected)
    radius_lookup = dict(zip(radius_edges, radius_segments, strict=True))
    mandatory.update(((edge, radius_lookup[edge]) for edge in selected_radius))
    mandatory_segments = list(mandatory.values())
    mandatory_index = _PeriodicBins(mandatory_segments, work)
    radius_lookup.clear()
    radius_edges.clear()
    radius_segments.clear()
    optional: list[Edge] = []
    optional_segments: list[_Segment] = []
    optional_penalties: list[tuple[int, int, RadicalElement]] = []
    for edge, segment, square in bank:
        if edge in mandatory:
            continue
        if any((_conflict(segment, mandatory_segments[index], grid, work) for index in mandatory_index.candidates(segment))):
            continue
        if not visible(edge, segment):
            continue
        optional.append(edge)
        optional_segments.append(segment)
        optional_penalties.append((int(square < radius2), int(square > radius2), square))
    edge_count = 3 * len(points)
    ranks = tuple((quantile * (edge_count - 1) for quantile in quantiles))
    below_limit = min((floor(rank) for rank in ranks))
    above_limit = edge_count - 1 - max((ceil(rank) for rank in ranks))
    mandatory_squares = {edge: square for edge, _, square in bank if edge in mandatory}
    for edge in mandatory:
        square = mandatory_squares[edge]
        below_limit -= int(square < radius2)
        above_limit -= int(square > radius2)
    mandatory_squares.clear()
    bank.clear()
    completion = _compatible_edges(optional, optional_segments, optional_penalties, grid, work, below_limit=below_limit, above_limit=above_limit)
    return tuple(sorted(set(mandatory) | set(completion)))

def _faces(edges: tuple[Edge, ...], points: tuple[Point, ...], grid: _Grid, work: _Predicates) -> list[list[Corner]]:
    half: list[Edge] = []
    outgoing: list[list[int]] = [[] for _ in points]
    directions: list[Point] = []
    for a, b, sx, sy in edges:
        delta = _sub(_lift(points[b], (sx, sy), grid.vectors), points[a])
        outgoing[a].append(len(half))
        half.append((a, b, sx, sy))
        directions.append(delta)
        outgoing[b].append(len(half))
        half.append((b, a, -sx, -sy))
        directions.append(_scale(delta, -1))

    def compare(i: int, j: int) -> int:
        a, b = (directions[i], directions[j])
        ah = a[1] > 0 or (a[1] == 0 and a[0] >= 0)
        bh = b[1] > 0 or (b[1] == 0 and b[0] >= 0)
        if ah != bh:
            return -1 if ah else 1
        cross = work.orient((RadicalElement(), RadicalElement()), a, b)
        if not cross:
            raise ValueError('periodic.algebraic.rays must have distinct directions; got coincident source graph rays')
        return -cross
    positions: dict[int, int] = {}
    for root, star in enumerate(outgoing):
        if len(star) < 2:
            raise ValueError(f'periodic.algebraic.completed_graph must include every constructed root; got root={root}, degree={len(star)}, edges={len(edges)}')
        star.sort(key=cmp_to_key(compare))
        positions.update(((edge, i) for i, edge in enumerate(star)))
    visited: set[int] = set()
    faces: list[list[Corner]] = []
    for start in range(len(half)):
        if start in visited:
            continue
        current, image = (start, (0, 0))
        face: list[Corner] = []
        while current not in visited:
            visited.add(current)
            a, b, sx, sy = half[current]
            face.append((a, image))
            image = (image[0] + sx, image[1] + sy)
            current = outgoing[b][(positions[current ^ 1] - 1) % len(outgoing[b])]
        if current != start or image != (0, 0) or len(face) < 3:
            raise ValueError('periodic.algebraic.faces must be contractible source polygons; got an open or winding face')
        faces.append(face)
    return faces

def _bind_materials(triangles: list[tuple[Corner, Corner, Corner]], points: tuple[Point, ...], grid: _Grid, work: _Predicates) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    source_triangles = [tuple((_lift(grid.points[int(root)], (int(image[0]), int(image[1])), grid.vectors) for root, image in zip(cell, images, strict=True))) for cell, images in zip(grid.strata.cells, grid.strata.shifts, strict=True)]
    source_fractional = [tuple((_fractional(point, grid) for point in triangle)) for triangle in source_triangles]
    regions: list[int] = []
    ancestry: list[tuple[int, ...]] = []
    for triangle in triangles:
        corners = tuple((_lift(points[root], image, grid.vectors) for root, image in triangle))
        fractional = tuple((_fractional(point, grid) for point in corners))
        members: set[int] = set()
        for parent, (source, sf) in enumerate(zip(source_triangles, source_fractional, strict=True)):
            ranges = tuple((range(ceil(min((p[k] for p in fractional)) - max((p[k] for p in sf))), floor(max((p[k] for p in fractional)) - min((p[k] for p in sf))) + 1) for k in range(2)))
            for sx, sy in product(ranges[0], ranges[1]):
                other = tuple((_lift(point, (sx, sy), grid.vectors) for point in source))
                separated = False
                for first, second in ((corners, other), (other, corners)):
                    if any((all((work.orient(first[k], first[(k + 1) % 3], p) <= 0 for p in second)) for k in range(3))):
                        separated = True
                        break
                if not separated:
                    members.add(parent)
        labels = {int(grid.strata.regions[parent]) for parent in members}
        if len(labels) != 1:
            raise ValueError('periodic.algebraic.material must resolve one source region per positive cell; got absent or crossed source strata')
        regions.append(next(iter(labels)))
        ancestry.append(tuple(sorted(members)))
    return (np.asarray(regions, dtype=np.int64), tuple(ancestry))

def _source_region_areas(grid: _Grid) -> dict[int, RadicalElement]:
    expected: dict[int, RadicalElement] = {}
    for cell, images, region in zip(grid.strata.cells, grid.strata.shifts, grid.strata.regions, strict=True):
        corners = tuple((_lift(grid.points[int(root)], (int(image[0]), int(image[1])), grid.vectors) for root, image in zip(cell, images, strict=True)))
        delta, other = (_sub(corners[1], corners[0]), _sub(corners[2], corners[0]))
        label = int(region)
        expected[label] = expected.get(label, RadicalElement()) + (delta[0] * other[1] - delta[1] * other[0]) / 2
    return expected

def _certify_carrier(triangles: list[tuple[Corner, Corner, Corner]], regions: np.ndarray, points: tuple[Point, ...], vectors: tuple[Point, Point], protected: tuple[Edge, ...], expected: dict[int, RadicalElement], work: _Predicates) -> None:
    """Prove positive cells, closed incidence, manifold links, and region areas."""
    incidence: dict[Edge, list[int]] = {}
    links: dict[int, list[tuple[Corner, Corner]]] = {}
    areas: dict[int, RadicalElement] = {}
    for triangle, region in zip(triangles, regions, strict=True):
        corners = tuple((_lift(points[root], image, vectors) for root, image in triangle))
        if work.orient(corners[0], corners[1], corners[2]) <= 0:
            raise ValueError('periodic.algebraic.orientation must be strictly positive; got a nonpositive completion cell')
        delta, other = (_sub(corners[1], corners[0]), _sub(corners[2], corners[0]))
        area = (delta[0] * other[1] - delta[1] * other[0]) / 2
        label = int(region)
        areas[label] = areas.get(label, RadicalElement()) + area
        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            root, image = triangle[a]
            first, fi = triangle[b]
            second, si = triangle[c]
            edge, sign = _key(root, image, first, fi)
            incidence.setdefault(edge, []).append(sign)
            links.setdefault(root, []).append(((first, (fi[0] - image[0], fi[1] - image[1])), (second, (si[0] - image[0], si[1] - image[1]))))
    if len(incidence) != 3 * len(points) or any((sorted(signs) != [-1, 1] for signs in incidence.values())) or (not set(protected) <= incidence.keys()):
        raise ValueError('periodic.algebraic.incidence must be closed and retain protected windings; got an invalid edge bank')
    for pairs in links.values():
        adjacency: dict[Corner, list[Corner]] = {}
        for a, b in pairs:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        if any((len(neighbors) != 2 for neighbors in adjacency.values())):
            raise ValueError('periodic.algebraic.vertex_link must be 2-regular; got a nonmanifold source-action vertex')
        pending, seen = ([min(adjacency)], set())
        while pending:
            corner = pending.pop()
            if corner not in seen:
                seen.add(corner)
                pending.extend(adjacency[corner])
        if len(seen) != len(adjacency):
            raise ValueError('periodic.algebraic.vertex_link must be connected; got multiple source-action link cycles')
    if areas != expected or len(links) != len(points):
        raise ValueError('periodic.algebraic.material_areas must equal the source partition and include every site; got an incomplete carrier')

@dataclass(frozen=True, slots=True)
class OrbitLatticeBands:
    """Exact admission proof of one orbit-lattice equilateral band carrier.

    The authored representatives are one coset of a translation lattice that
    contains the cell lattice. ``step`` is a primitive vector of that lattice whose
    exact length is the hard target; ``rise`` completes a positively oriented basis
    with ``0 <= step.rise < |step|^2``. Every root receives one equilateral apex on
    the ``rise`` side, and ``lead`` names the next-row seed ahead of that apex. The
    authoritative zero-tolerance hard verdict passed on the complete closed-form
    edge bank of this carrier.
    """
    roots: tuple[_Vector, ...]
    fractional: tuple[_Vector, ...]
    residues: Mapping[_Vector, int]
    vectors: tuple[_Vector, _Vector]
    dual: tuple[_Vector, _Vector]
    step: _Vector
    rise: _Vector
    lead: int
    target: Fraction

def _hard_uniform_controls(specification: SurfaceMeshingSpec | VolumeMeshingSpec) -> list[UniformSizeControl] | None:
    hard: list[UniformSizeControl] = []
    for control in specification.size_controls:
        match control:
            case UniformSizeControl() if control.strength is SizeControlStrength.HARD:
                hard.append(control)
            case UniformSizeControl():
                pass
            case _:
                return None
    return hard

def _dot(a: _Vector, b: _Vector) -> Fraction:
    return a[0] * b[0] + a[1] * b[1]

def _cross(a: _Vector, b: _Vector) -> Fraction:
    return a[0] * b[1] - a[1] * b[0]

def _combine(m: Fraction | int, a: _Vector, k: Fraction | int, b: _Vector) -> _Vector:
    return (m * a[0] + k * b[0], m * a[1] + k * b[1])

def _rational_fractional(point: _Vector, dual: tuple[_Vector, _Vector]) -> _Vector:
    return (_dot(dual[0], point), _dot(dual[1], point))

def _integer_lattice_basis(generators: tuple[tuple[int, int], ...], work: _Predicates) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the Hermite basis ``((alpha, 0), (beta, gamma))`` of a full-rank span."""
    pivot_x = pivot_y = alpha = 0
    for x, y in generators:
        while y:
            quotient = pivot_y // y
            pivot_x, pivot_y, x, y = (x, y, pivot_x - quotient * x, pivot_y - quotient * y)
        alpha = gcd(alpha, x)
    if pivot_y < 0:
        pivot_x, pivot_y = (-pivot_x, -pivot_y)
    return ((alpha, 0), (pivot_x % alpha, pivot_y))

def _reduced_basis(first: _Vector, second: _Vector, work: _Predicates) -> tuple[_Vector, _Vector]:
    """Lagrange-reduce a rational lattice basis and orient it positively."""
    while True:
        if _dot(first, first) > _dot(second, second):
            first, second = (second, first)
        quotient = round(_dot(first, second) / _dot(first, first))
        if not quotient:
            break
        second = _combine(1, second, -quotient, first)
    if _cross(first, second) < 0:
        second = (-second[0], -second[1])
    return (first, second)

def _bezout(a: int, b: int) -> tuple[int, int]:
    old_r, r, old_s, s, old_t, t = (a, b, 1, 0, 0, 1)
    while r:
        quotient = old_r // r
        old_r, r = (r, old_r - quotient * r)
        old_s, s = (s, old_s - quotient * s)
        old_t, t = (t, old_t - quotient * t)
    return (-old_s, -old_t) if old_r < 0 else (old_s, old_t)

def _target_steps(first: _Vector, second: _Vector, target: Fraction, work: _Predicates) -> list[tuple[_Vector, _Vector, int]]:
    """Enumerate primitive lattice vectors of exact target length with complements.

    The reduced Gram form bounds both coefficients, so every vector of length
    ``target`` is visited. Each step keeps one sign; its complement is normalized
    to ``0 <= step.rise < target^2`` and ``lead`` selects the next-row seed whose
    projection follows the apex ``step / 2``.
    """
    square = target * target
    g11, g12, g22 = (_dot(first, first), _dot(first, second), _dot(second, second))
    bound = isqrt(floor(square * g11 / (g11 * g22 - g12 * g12)))
    steps: list[tuple[_Vector, _Vector, int]] = []
    for k in range(-bound, bound + 1):
        discriminant = (g12 * k) ** 2 - g11 * (g22 * k * k - square)
        if discriminant < 0:
            continue
        radius = isqrt(floor(discriminant / (g11 * g11))) + 1
        center = -g12 * k / g11
        for m in range(floor(center) - radius, ceil(center) + radius + 1):
            if gcd(m, k) != 1 or g11 * m * m + 2 * g12 * m * k + g22 * k * k != square:
                continue
            step = _combine(m, first, k, second)
            if step[1] < 0 or (step[1] == 0 and step[0] < 0):
                continue
            x, y = _bezout(m, k)
            rise = _combine(-y, first, x, second)
            rise = _combine(1, rise, -floor(_dot(step, rise) / square), step)
            steps.append((step, rise, int(2 * _dot(step, rise) < square)))
    return sorted(steps, key=lambda item: (item[0][1], item[0][0]))

def _band_apex(step: _Vector, predicates: _Predicates) -> Point:
    height = RadicalElement.sqrt(Fraction(3, 4))
    return (RadicalElement(step[0] / 2) - height * step[1], RadicalElement(step[1] / 2) + height * step[0])

def _orbit_root(proof: OrbitLatticeBands, point: _Vector, work: _Predicates) -> Corner:
    """Name the seed orbit and lattice image of a point of the admitted coset."""
    fractional = _rational_fractional(point, proof.dual)
    root = proof.residues[fractional[0] - floor(fractional[0]), fractional[1] - floor(fractional[1])]
    base = proof.fractional[root]
    image = (fractional[0] - base[0], fractional[1] - base[1])
    if image[0].denominator != 1 or image[1].denominator != 1:
        raise ValueError('periodic.algebraic.orbit_lattice.image must be an integer lattice action; got a point outside the admitted coset')
    return (root, (int(image[0]), int(image[1])))

def _band_neighbors(proof: OrbitLatticeBands, root: int, work: _Predicates) -> tuple[Corner, Corner, Corner, Corner]:
    """Return the next, previous, leading and trailing next-row seed corners."""
    point, step, rise = (proof.roots[root], proof.step, proof.rise)
    return (_orbit_root(proof, _combine(1, point, 1, step), work), _orbit_root(proof, _combine(1, point, -1, step), work), _orbit_root(proof, _combine(1, _combine(1, point, proof.lead, step), 1, rise), work), _orbit_root(proof, _combine(1, _combine(1, point, proof.lead - 1, step), 1, rise), work))

def prove_orbit_lattice_bands(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
) -> OrbitLatticeBands | None:
    """Admit equilateral row bands only from exact seed and request premises.

    The premises are a zero-tolerance hard uniform target, seed representatives
    exhausting one coset of a translation lattice, equilateral apexes strictly
    between consecutive seed rows, and a passing exact hard verdict of the
    closed-form edge bank. ``None`` means a premise does not hold; the construction is not admitted.
    """
    match source.domain:
        case PeriodicPointOrbits() as orbits:
            pass
        case CellMesh():
            return None
        case _:
            return None
    policy = specification.size_compliance
    controls = _hard_uniform_controls(specification)
    if (
        source.ambient_dimension != 2
        or source.site_weights is not None
        or bool(specification.protected_features)
        or bool(specification.region_controls)
        or bool(specification.patch_controls)
        or bool(specification.layer_controls)
        or policy.absolute_tolerance != 0
        or policy.relative_tolerance != 0
        or not controls
    ):
        return None
    target = Fraction(float(controls[0].target_size))
    if any(
        Fraction(float(control.target_size)) != target
        or control.maximum_growth_rate is not None
        for control in controls
    ):
        return None
    raw = np.asarray(orbits.cell.vectors, dtype=np.float64)
    vectors = (
        (Fraction(float(raw[0, 0])), Fraction(float(raw[0, 1]))),
        (Fraction(float(raw[1, 0])), Fraction(float(raw[1, 1]))),
    )
    a, b = vectors
    determinant = _cross(a, b)
    if determinant <= 0:
        return None
    dual = (
        (b[1] / determinant, -b[0] / determinant),
        (-a[1] / determinant, a[0] / determinant),
    )
    ids = np.asarray(orbits.point_ids, dtype=np.int64)
    if any(control.scope.source_id != source.source_id or control.scope.source_revision != source.source_revision or control.scope.entity_dimension != 0 or control.scope.entity_set_id != source.scope_entity_set_id(0) or set(map(int, control.scope.entity_ids)) != set(map(int, ids)) for control in controls):
        return None
    order = np.argsort(ids, kind="stable")
    representatives = np.asarray(orbits.points, dtype=np.float64)[order]
    count = representatives.shape[0]
    if count == 0:
        return None
    work = _Predicates()
    roots = tuple(
        (Fraction(float(x)), Fraction(float(y))) for x, y in representatives.tolist()
    )
    fractional = tuple(_rational_fractional(root, dual) for root in roots)
    residues: dict[_Vector, int] = {}
    for index, value in enumerate(fractional):
        residues.setdefault(
            (value[0] - floor(value[0]), value[1] - floor(value[1])), index
        )
    if len(residues) != count:
        return None
    origin = fractional[0]
    differences = tuple(
        (value[0] - origin[0], value[1] - origin[1]) for value in fractional[1:]
    )
    denominator = lcm(1, *(part.denominator for value in differences for part in value))
    (alpha, _), (beta, gamma) = _integer_lattice_basis(
        (
            (denominator, 0),
            (0, denominator),
            *(
                (int(value[0] * denominator), int(value[1] * denominator))
                for value in differences
            ),
        ),
        work,
    )
    # Distinct roots in one coset exhaust it exactly when they meet its index.
    if denominator * denominator != count * alpha * gamma:
        return None
    # The apex height sqrt(3)/2*target lies strictly below the row spacing.
    if 3 * target**4 >= 4 * (determinant / count) ** 2:
        return None
    first, second = _reduced_basis(
        _combine(Fraction(alpha, denominator), a, 0, b),
        _combine(Fraction(beta, denominator), a, Fraction(gamma, denominator), b),
        work,
    )
    for step, rise, lead in _target_steps(first, second, target, work):
        proof = OrbitLatticeBands(
            roots, fractional, residues, vectors, dual, step, rise, lead, target
        )
        if _orbit_size_ok(proof, controls, policy, work):
            return proof
    return None






def _size_ok(squares: list[RadicalElement], controls: list[UniformSizeControl], policy: SizeCompliancePolicy) -> bool:
    if not squares:
        return False
    ordered = sorted(squares)
    for control in controls:
        target = Fraction(float(control.target_size))
        if control.minimum_size is not None and ordered[0] < Fraction(float(control.minimum_size)) ** 2:
            return False
        if control.maximum_size is not None and ordered[-1] > Fraction(float(control.maximum_size)) ** 2:
            return False
        for name in policy.target_statistics:
            rank = (Fraction(1, 2) if name == "p50" else Fraction(19, 20)) * (len(ordered) - 1)
            low = floor(rank)
            high = ceil(rank)
            weight = rank - low
            x, y = ordered[low], ordered[high]
            if not weight:
                if x != target * target:
                    return False
            else:
                a, b = 1 - weight, weight
                residual = RadicalElement(target * target) - a * a * x - b * b * y
                if residual < 0 or residual * residual != 4 * a * a * b * b * x * y:
                    return False
    return True


def _orbit_size_ok(proof: OrbitLatticeBands, controls: list[UniformSizeControl], policy: SizeCompliancePolicy, work: _Predicates) -> bool:
    apex = _band_apex(proof.step, work)
    def filler(shift: int) -> RadicalElement:
        offset = _combine(shift, proof.step, 1, proof.rise)
        x, y = offset[0] - apex[0], offset[1] - apex[1]
        return x * x + y * y
    values = [RadicalElement(proof.target ** 2)] * 4 + [filler(proof.lead), filler(proof.lead - 1)]
    return _size_ok(values * len(proof.roots), controls, policy)


def _carrier_squares(triangles: list[tuple[Corner, Corner, Corner]], points: tuple[Point, ...], vectors: tuple[Point, Point]) -> list[RadicalElement]:
    squares: dict[Edge, RadicalElement] = {}
    for triangle in triangles:
        for (first, fi), (second, si) in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge, _ = _key(first, fi, second, si)
            if edge not in squares:
                squares[edge] = _square(points[edge[0]], _lift(points[edge[1]], (edge[2], edge[3]), vectors))
    return list(squares.values())


def _finish_carrier(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec,
    coordinates: Sequence[Point],
    triangles: list[tuple[Corner, Corner, Corner]],
    regions: np.ndarray,
    ancestry: tuple[tuple[int, ...], ...],
    identities: Sequence[SiteIdentity],
    kind: str,
) -> ExactAlgebraicCarrier:
    points = tuple(coordinates)
    rounded = np.asarray([[value.rounded()[0] for value in point] for point in points], dtype=np.float64)
    simplices = np.asarray([[root for root, _ in triangle] for triangle in triangles], dtype=np.int64)
    shifts = np.asarray([[image for _, image in triangle] for triangle in triangles], dtype=np.int64)
    for array in (rounded, simplices, shifts, regions):
        array.setflags(write=False)
    identity = canonical_fingerprint({"kind": kind, "source_binding": source.binding_id, "policy": specification.size_compliance.policy_id, "controls": [control.control_id for control in specification.size_controls], "sites": identities})
    return ExactAlgebraicCarrier(points, rounded, simplices, shifts, regions, ancestry, tuple(identities), identity)


def prove_source_action_bands(source: NativePeriodicSource, specification: SurfaceMeshingSpec, /) -> bool:
    """Admit the material construction only from the represented source actions."""
    return _recover_grid(source, specification, _Predicates()) is not None


def construct_exact_algebraic_carrier(source: NativePeriodicSource, specification: SurfaceMeshingSpec, /) -> ExactAlgebraicCarrier:
    """Construct and certify protected material source-action four-band cells."""
    work = _Predicates()
    grid = _recover_grid(source, specification, work)
    if grid is None:
        raise ValueError("Source-action four-band construction premises are not satisfied.")
    coordinates, identities = _sites(grid, work)
    quantiles = tuple(Fraction(1, 2) if name == "p50" else Fraction(19, 20) for name in specification.size_compliance.target_statistics)
    edges = _target_graph_search(coordinates, grid, work, quantiles)
    faces = _faces(edges, coordinates, grid, work)
    if len(faces) != 2 * len(coordinates) or any(len(face) != 3 for face in faces):
        raise ValueError("Source-action completion must be a closed triangulation.")
    triangles = [(face[0], face[1], face[2]) for face in faces]
    regions, ancestry = _bind_materials(triangles, coordinates, grid, work)
    _certify_carrier(triangles, regions, coordinates, grid.vectors, grid.protected, _source_region_areas(grid), work)
    controls = _hard_uniform_controls(specification)
    if controls is None or not _size_ok(_carrier_squares(triangles, coordinates, grid.vectors), controls, specification.size_compliance):
        raise ValueError("Source-action carrier violates exact hard size controls.")
    return _finish_carrier(source, specification, coordinates, triangles, regions, ancestry, identities, "source-action-four-band-algebraic-carrier")


def construct_orbit_lattice_band_carrier(source: NativePeriodicSource, specification: SurfaceMeshingSpec, /) -> ExactAlgebraicCarrier:
    """Construct and certify exact equilateral orbit bands and filler cells."""
    proof = prove_orbit_lattice_bands(source, specification)
    controls = _hard_uniform_controls(specification)
    if proof is None or not controls:
        raise ValueError("Orbit-lattice band construction premises are not satisfied.")
    domain = source.domain
    if not isinstance(domain, PeriodicPointOrbits):
        raise ValueError("Orbit-lattice band construction requires periodic point orbits.")
    work = _Predicates()
    roots = len(proof.roots)
    a, b = proof.vectors
    vectors = ((RadicalElement(a[0]), RadicalElement(a[1])), (RadicalElement(b[0]), RadicalElement(b[1])))
    apex = _band_apex(proof.step, work)
    coordinates = [(RadicalElement(x), RadicalElement(y)) for x, y in proof.roots]
    ids = sorted(map(int, domain.point_ids))
    identities = [("authored_source_root", (identifier,)) for identifier in ids]
    wraps: list[Image] = []
    for root in range(roots):
        raw = _add(coordinates[root], apex)
        fractional = _fractional(raw, proof)
        image = (floor(fractional[0]), floor(fractional[1]))
        coordinates.append(_lift(raw, (-image[0], -image[1]), vectors))
        wraps.append(image)
        identities.append(("orbit_lattice_band_apex", (ids[root],)))
    def apex_corner(corner: Corner) -> Corner:
        root, image = corner
        return roots + root, (image[0] + wraps[root][0], image[1] + wraps[root][1])
    triangles: list[tuple[Corner, Corner, Corner]] = []
    for root in range(roots):
        right, left, ahead, behind = _band_neighbors(proof, root, work)
        here = (root, (0, 0))
        top = apex_corner(here)
        triangles.extend(((here, right, top), (here, top, apex_corner(left)), (top, apex_corner(right), ahead), (top, ahead, behind)))
    points = tuple(coordinates)
    regions = np.zeros(len(triangles), dtype=np.int64)
    _certify_carrier(triangles, regions, points, vectors, (), {0: RadicalElement(_cross(a, b))}, work)
    if not _size_ok(_carrier_squares(triangles, points, vectors), controls, specification.size_compliance):
        raise ValueError("Orbit-lattice carrier violates exact hard size controls.")
    return _finish_carrier(source, specification, points, triangles, regions, tuple((index,) for index in range(len(triangles))), identities, "orbit-lattice-equilateral-band-algebraic-carrier")


__all__ = ["ExactAlgebraicCarrier", "OrbitLatticeBands", "prove_source_action_bands", "construct_exact_algebraic_carrier", "prove_orbit_lattice_bands", "construct_orbit_lattice_band_carrier", "periodic_quotient_edge_squared_lengths"]
