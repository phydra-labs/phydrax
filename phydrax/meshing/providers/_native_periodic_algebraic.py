#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Construct algebraic periodic carriers admitted from exact source premises.

Represented source-action grids receive four-band carriers selected by source
incidence, deck actions, material sides, and a hard size request compared in the
coefficient field; their topology is a maximum compatible radius-edge graph with a
finite planar completion. Candidate banks, bigint graphs and search frontiers are
admitted by live demand; closed rational periodic bins only prune disjoint pairs.
Translational seed orbits forming one lattice coset
receive equilateral row bands whose closed-form edge bank is admitted by the
exact hard verdict before construction.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from itertools import product
from math import ceil, floor, gcd, isqrt, lcm
from typing import assert_never, final

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import NativeExecutionScope
from ..._resource_ticket import HostBatchTicket
from ..._strict import StrictModule
from ...discretization._cell_mesh import CellMesh
from ...discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
    CoordinateLiveStorage,
)
from ...discretization._exact_algebraic_geometry import RadicalElement
from ...discretization._periodic_cell import PeriodicCell
from ...typing import checked
from .. import _periodic as periodic_owner
from .._contracts import MeshingLimits, SurfaceMeshingSpec, VolumeMeshingSpec
from .._controls import FeatureKind
from .._failures import MeshingFailure, MeshingFailureCategory
from .._sizing import (
    size_statistic_quantile,
    SizeCompliancePolicy,
    SizeControlStrength,
    UniformSizeControl,
)
from . import _native_periodic as source_owner
from ._native_periodic_constraints import (
    PeriodicSourceStrata,
    prepare_periodic_source_strata,
)


type Point = tuple[RadicalElement, RadicalElement]
type Image = tuple[int, int]
type Edge = tuple[int, int, int, int]
type Corner = tuple[int, Image]
type SiteIdentity = tuple[str, tuple[int, ...]]


def _bitset_bytes(width: int) -> int:
    # CPython stores a nonnegative bigint in 30-bit, four-byte digits, not bytes
    # of a packed C bit array. Include its header and allocation alignment.
    return 32 + 4 * ((width + 29) // 30)


def _scalar_bytes(value: RadicalElement) -> int:
    return 128 + sum(
        192
        + _bitset_bytes(radicand.bit_length())
        + _bitset_bytes(abs(coefficient.numerator).bit_length())
        + _bitset_bytes(coefficient.denominator.bit_length())
        for radicand, coefficient in value.terms
    )


def _point_bytes(point: Point) -> int:
    return 64 + sum(_scalar_bytes(value) for value in point)


def _box_bytes(
    a: Point,
    b: Point,
    box: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
) -> int:
    return 512 + _point_bytes(a) + _point_bytes(b) + sum(
        64
        + _bitset_bytes(abs(value.numerator).bit_length())
        + _bitset_bytes(value.denominator.bit_length())
        for interval in box
        for value in interval
    )


@final
class ExactAlgebraicCarrier(StrictModule):
    """Retain computed source identities, field-valued geometry, and charged work."""

    points: np.ndarray
    exact_coordinates: tuple[Point, ...] = eqx.field(static=True)
    simplices: np.ndarray
    shifts: np.ndarray
    regions: np.ndarray
    ancestry: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    work: int = eqx.field(static=True)
    image_count: int = eqx.field(static=True)
    source_identity: str = eqx.field(static=True)
    site_identities: tuple[SiteIdentity, ...] = eqx.field(static=True)
    work_evidence: tuple[tuple[str, int], ...] = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class _Grid:
    strata: PeriodicSourceStrata
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


class _Work:
    """Charge each finite search operation before performing its arithmetic."""

    def __init__(self, budget: CoordinateEnclosureBudget) -> None:
        self.budget = budget
        self.counts: dict[str, int] = {}
        self.boxes: dict[
            tuple[Point, Point],
            tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
        ] = {}
        self._box_storage: CoordinateLiveStorage | None = None

    @contextmanager
    def box_scope(self) -> Iterator[None]:
        """Keep cached enclosures live while releasing each interval workspace."""
        if self._box_storage is not None:
            yield
            return
        with self.budget.live_storage() as storage:
            storage.set_bound(256)
            self._box_storage = storage
            try:
                yield
            finally:
                self.boxes.clear()
                self._box_storage = None

    def box(
        self, a: Point, b: Point
    ) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
        key = (a, b)
        old = self.boxes.get(key)
        if old is not None:
            return old
        storage = self._box_storage
        if storage is None:
            raise RuntimeError("periodic.algebraic.box requires an active box scope")
        # Interval evaluation owns scratch; only the four selected rational
        # endpoints and their cache entry survive this scope.
        with self.budget.temporary_scope():
            self.budget.reserve(0, 512)
            ai = tuple(value.interval(bits=32, budget=self.budget) for value in a)
            bi = tuple(value.interval(bits=32, budget=self.budget) for value in b)
            result = (
                (min(ai[0][0], bi[0][0]), max(ai[0][1], bi[0][1])),
                (min(ai[1][0], bi[1][0]), max(ai[1][1], bi[1][1])),
            )
            storage.set_bound(storage.bound + _box_bytes(a, b, result))
            self.boxes[key] = result
        return result

    def discard_boxes_except(self, segments: list[_Segment]) -> None:
        """Drop candidate enclosures after their final visibility consumer."""
        storage = self._box_storage
        if storage is None:
            raise RuntimeError("periodic.algebraic.box requires an active box scope")
        with self.budget.temporary_scope():
            self.budget.reserve(0, 512 + 512 * len(segments))
            keep = {(segment.fa, segment.fb) for segment in segments}
            self.spend("periodic_box_retention_visits", len(self.boxes))
            # Rebuild rather than delete: the old dictionary's capacity and
            # discarded coefficient keys must actually die before release.
            self.boxes = {key: box for key, box in self.boxes.items() if key in keep}
            storage.set_bound(
                256 + sum(_box_bytes(*key, box) for key, box in self.boxes.items())
            )

    def spend(self, kind: str, count: int = 1) -> None:
        self.budget.reserve(count)
        self.counts[kind] = self.counts.get(kind, 0) + count

    def orient(self, a: Point, b: Point, c: Point) -> int:
        self.spend("orientation_predicates")
        return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).sign()


@contextmanager
def _phase(
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    maximum_work: int,
    maximum_bytes: int,
) -> Iterator[None]:
    if not budget.ticket.settled:
        with (
            budget.ticket.native_target(),
            budget.bound_stage(maximum_work, maximum_bytes),
        ):
            yield
        return
    remaining = scope.remaining()
    ticket = scope.batch(
        min(maximum_work, remaining.remaining_work_units),
        queries_upper=remaining.remaining_geometry_queries,
        bytes_upper=min(maximum_bytes, remaining.remaining_managed_bytes),
        cavity_upper=remaining.remaining_cavity_cells,
    )
    with (
        budget.bind_ticket(ticket),
        ticket.native_target(),
        budget.bound_stage(maximum_work, maximum_bytes),
    ):
        yield


def _key(a: int, ia: Image, b: int, ib: Image) -> tuple[Edge, int]:
    if (a, *ia) <= (b, *ib):
        return (a, b, ib[0] - ia[0], ib[1] - ia[1]), 1
    return (b, a, ia[0] - ib[0], ia[1] - ib[1]), -1


def _add(a: Point, b: Point) -> Point:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: Point, b: Point) -> Point:
    return a[0] - b[0], a[1] - b[1]


def _scale(a: Point, factor: RadicalElement | Fraction | int) -> Point:
    return a[0] * factor, a[1] * factor


def _square(a: Point, b: Point) -> RadicalElement:
    x, y = _sub(b, a)
    return x * x + y * y


def _lift(point: Point, image: Image, vectors: tuple[Point, Point]) -> Point:
    return _add(point, _add(_scale(vectors[0], image[0]), _scale(vectors[1], image[1])))


def _fractional(point: Point, grid: _Grid | OrbitLatticeBands) -> Point:
    first, second = grid.dual
    return (
        point[0] * first[0] + point[1] * first[1],
        point[0] * second[0] + point[1] * second[1],
    )


def _wrap(point: Point, grid: _Grid) -> Point:
    fractional = _fractional(point, grid)
    return _lift(point, (-floor(fractional[0]), -floor(fractional[1])), grid.vectors)


def _recover_grid(
    source: source_owner.NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    work: _Work,
) -> _Grid | None:
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
    topology = periodic_owner.require_periodic_topology(mesh)
    match topology.cell:
        case PeriodicCell():
            pass
        case _:
            return None
    policy = specification.size_compliance
    if (
        policy.absolute_tolerance != 0
        or policy.relative_tolerance != 0
        or set(policy.target_statistics) != {"p50", "p95"}
    ):
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
        for feature in specification.protected_features
    ):
        return None
    count = np.unique(np.asarray(topology.vertex_representatives)).size
    n = isqrt(count)
    if n < 4 or n % 2 or n * n != count:
        return None
    # Admit source arrays and discrete incidence before host preparation. The
    # larger topology bank is admitted only after eligibility is established.
    work.budget.reserve(0, 1024 * count + 4096 * mesh.blocks[0].cell_count)
    work.spend("source_incidence", 3 * mesh.blocks[0].cell_count)
    strata = prepare_periodic_source_strata(
        mesh, source.cell_regions, specification.protected_features
    )
    raw_vectors = np.asarray(strata.cell.vectors)
    rational_vectors = (
        (Fraction(float(raw_vectors[0, 0])), Fraction(float(raw_vectors[0, 1]))),
        (Fraction(float(raw_vectors[1, 0])), Fraction(float(raw_vectors[1, 1]))),
    )
    a, b = rational_vectors
    determinant = a[0] * b[1] - a[1] * b[0]
    norm = a[0] * a[0] + a[1] * a[1]
    if determinant <= 0:
        return None
    radius = Fraction(float(controls[0].target_size))
    if radius * radius != norm / (n * n):
        return None
    material_ids = set(int(value) for value in strata.regions)
    if len(material_ids) != 2 or any(
        Fraction(float(control.target_size)) != radius
        or control.minimum_size is not None
        or control.maximum_growth_rate is not None
        or control.scope.entity_dimension != 2
        or set(int(value) for value in control.scope.entity_ids) != material_ids
        for control in controls
    ):
        return None
    if any(control.maximum_size is None for control in controls):
        return None
    maximum = min(
        Fraction(float(control.maximum_size))
        for control in controls
        if control.maximum_size is not None
    )
    if maximum < radius or maximum > 3 * radius / 2:
        return None
    dual = (
        (b[1] / determinant, -b[0] / determinant),
        (-a[1] / determinant, a[0] / determinant),
    )
    # This strict bound proves short edges cannot be nontrivial self-orbits.
    if any(maximum * (abs(row[0]) + abs(row[1])) >= Fraction(1, 2) for row in dual):
        return None
    points = tuple(
        (RadicalElement(Fraction(float(row[0]))), RadicalElement(Fraction(float(row[1]))))
        for row in strata.points
    )
    vectors = (
        (RadicalElement(a[0]), RadicalElement(a[1])),
        (RadicalElement(b[0]), RadicalElement(b[1])),
    )
    incidence_lists: dict[Edge, list[tuple[int, int]]] = {}
    actions: list[dict[int, tuple[int, Image]]] = [{}, {}]
    for ci, (cell, images) in enumerate(zip(strata.cells, strata.shifts, strict=True)):
        corners = [
            _lift(points[int(root)], (int(image[0]), int(image[1])), vectors)
            for root, image in zip(cell, images, strict=True)
        ]
        if work.orient(*corners) <= 0:
            return None
        for first, second in ((0, 1), (1, 2), (2, 0)):
            ia = (int(images[first, 0]), int(images[first, 1]))
            ib = (int(images[second, 0]), int(images[second, 1]))
            edge, sign = _key(int(cell[first]), ia, int(cell[second]), ib)
            incidence_lists.setdefault(edge, []).append((ci, sign))
    if any(
        len(members) != 2 or sum(sign for _, sign in members) != 0
        for members in incidence_lists.values()
    ):
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
                    actions[axis][root] = target, image
    if any(len(action) != count for action in actions):
        return None
    anchor = min(actions[0])
    rows: list[tuple[int, ...]] = []
    vertical: list[Edge] = []
    root = anchor
    column_image = (0, 0)
    seen: set[int] = set()
    for column in range(n):
        row_roots: list[int] = []
        current, winding = root, (0, 0)
        for row in range(n):
            work.spend("action_grid_steps")
            if current in seen:
                return None
            seen.add(current)
            row_roots.append(current)
            target, shift = actions[1][current]
            vertical.append(_key(current, (0, 0), target, shift)[0])
            # Commuting primitive deck actions, not incidental vertex order.
            xroot, ximage = actions[0][current]
            yxroot, yximage = actions[1][xroot]
            xyroot, xyimage = actions[0][target]
            if yxroot != xyroot or tuple(
                ximage[k] + yximage[k] for k in range(2)
            ) != tuple(shift[k] + xyimage[k] for k in range(2)):
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
    protected = tuple(
        (int(row[0]), int(row[1]), int(row[2]), int(row[3])) for row in strata.edges
    )
    material_edges = {
        edge
        for edge, members in incidence_lists.items()
        if len({int(strata.regions[ci]) for ci, _ in members}) == 2
    }
    if (
        set(protected) != material_edges
        or not material_edges <= set(vertical)
        or len(protected) != 2 * n
    ):
        return None
    components = [
        frozenset(row)
        for column, row in enumerate(rows)
        if all(edge in material_edges for edge in vertical[column * n : (column + 1) * n])
    ]
    if len(components) != 2:
        return None
    component_columns = [i for i, row in enumerate(rows) if frozenset(row) in components]
    if (component_columns[1] - component_columns[0]) % n != n // 2:
        return None
    # The profile is the small-height intersection of r²+t²=h² and
    # C*r+H*t=(8*h²+C²+H²)/6, where C is the source shear plus two h edges.
    shear = (a[0] * b[0] + a[1] * b[1]) / norm
    aspect = determinant / norm
    if not 0 < shear < Fraction(1, 2) or shear * shear + aspect * aspect >= 4:
        return None
    h, height_total = Fraction(1, n), aspect / n
    closure = shear / n + 2 * h
    length2 = closure * closure + height_total * height_total
    intercept = (8 * h * h + length2) / 6
    radicand = h * h * length2 - intercept * intercept
    if closure <= 0 or height_total <= 0 or radicand <= 0:
        return None
    radical = RadicalElement.sqrt(radicand, budget=work.budget)
    run = (intercept * closure + height_total * radical) / length2
    height = (intercept * height_total - closure * radical) / length2
    if not (
        0 < height < height_total / 4
        and run > 0
        and 3 * run - closure > 0
        and height_total - 3 * height > 0
    ):
        return None
    return _Grid(
        strata,
        points,
        vectors,
        dual,
        tuple(rows),
        tuple(vertical),
        {edge: tuple(members) for edge, members in incidence_lists.items()},
        protected,
        max(components, key=min),
        min(material_ids),
        radius,
        maximum,
        run,
        height,
    )


def _sites(
    grid: _Grid, work: _Work
) -> tuple[tuple[Point, ...], tuple[SiteIdentity, ...]]:
    points = list(grid.points)
    identities: list[SiteIdentity] = [
        ("authored_source_root", (root,)) for root in range(len(points))
    ]
    normal = (-grid.vectors[0][1], grid.vectors[0][0])
    step = _add(_scale(grid.vectors[0], grid.run), _scale(normal, grid.height))
    n = len(grid.rows)
    for row in range(n):
        for layer in range(1, 4):
            for column in range(n):
                work.spend("band_sites")
                root = grid.rows[column][row]
                edge = grid.vertical[column * n + row]
                points.append(_wrap(_add(grid.points[root], _scale(step, layer)), grid))
                identities.append(("source_edge_band", (*edge, root, layer)))
    for edge in grid.protected:
        first, second, sx, sy = edge
        a, b = grid.points[first], _lift(grid.points[second], (sx, sy), grid.vectors)
        delta = _sub(b, a)
        square = _square(a, b)
        multiplier2 = grid.radius * grid.radius / square - Fraction(1, 4)
        if multiplier2 <= 0:
            raise ValueError(
                "periodic.algebraic.protected_apex must have two circle intersections; got an invalid source edge"
            )
        # Authored source segments have rational coordinates. Extract their
        # rational coefficient before forming the algebraic circle radical.
        if len(multiplier2.terms) != 1 or multiplier2.terms[0][0] != 1:
            raise ValueError(
                "periodic.algebraic.source_edge must be rational; got a radical source edge"
            )
        multiplier = RadicalElement.sqrt(multiplier2.terms[0][1], budget=work.budget)
        for side in (-1, 1):
            side_region = next(
                int(grid.strata.regions[ci])
                for ci, sign in grid.incidence[edge]
                if sign == side
            )
            if first in grid.omitted_component and side_region == grid.omitted_region:
                continue
            work.spend("protected_circle_sites")
            point = _add(
                _scale(_add(a, b), Fraction(1, 2)),
                _scale((-delta[1], delta[0]), side * multiplier),
            )
            points.append(_wrap(point, grid))
            identities.append(
                ("protected_edge_side_material", (*edge, side, side_region))
            )
    if len(set(points)) != len(points):
        raise ValueError(
            "periodic.algebraic.sites must have distinct field-valued orbit coordinates; got coincident constructed sites"
        )
    return tuple(points), tuple(identities)


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


def _segment_bytes(segment: _Segment) -> int:
    return 128 + sum(
        _point_bytes(point) for point in (segment.a, segment.b, segment.fa, segment.fb)
    )


class _PeriodicBins:
    """Index closed rational enclosures on the unit torus without image banks."""

    def __init__(self, segments: list[_Segment], work: _Work) -> None:
        self.work = work
        self.segments = segments
        self.width = max(1, isqrt(len(segments)))
        self.mask_bytes = _bitset_bytes(len(segments))
        work.budget.reserve(0, 512 + 4 * self.mask_bytes)
        self.bins: dict[tuple[int, int], int] = {}
        for index, segment in enumerate(segments):
            bit = 1 << index
            for key in self._keys(segment):
                work.spend("periodic_bin_insertions")
                if key not in self.bins:
                    work.budget.reserve(0, 256 + self.mask_bytes)
                self.bins[key] = self.bins.get(key, 0) | bit

    def _keys(self, segment: _Segment) -> Iterator[tuple[int, int]]:
        box = self.work.box(segment.fa, segment.fb)
        first, second = (
            range(
                floor(lo * self.width),
                min(floor(hi * self.width) + 1, floor(lo * self.width) + self.width),
            )
            for lo, hi in box
        )
        for x in first:
            for y in second:
                yield x % self.width, y % self.width

    def candidates(self, segment: _Segment, *, stop: int | None = None) -> Iterator[int]:
        mask = 0
        for key in self._keys(segment):
            self.work.spend("periodic_bin_queries")
            mask |= self.bins.get(key, 0)
        if stop is not None:
            mask &= (1 << stop) - 1
        box = self.work.box(segment.fa, segment.fb)
        # Ascending low bits preserve the original exhaustive predicate order.
        while mask:
            bit = mask & -mask
            index = bit.bit_length() - 1
            mask ^= bit
            other = self.segments[index]
            other_box = self.work.box(other.fa, other.fb)
            self.work.spend("periodic_box_pairs")
            if all(
                ceil(box[k][0] - other_box[k][1])
                <= floor(box[k][1] - other_box[k][0])
                for k in range(2)
            ):
                yield index


def _conflict_graph(
    segments: list[_Segment], grid: _Grid, work: _Work
) -> list[int]:
    count = len(segments)
    work.budget.reserve(0, 256 + count * (16 + _bitset_bytes(count)))
    adjacency = [0] * count
    with work.budget.temporary_scope():
        index = _PeriodicBins(segments, work)
        for i, segment in enumerate(segments):
            for j in index.candidates(segment, stop=i):
                if _conflict(segment, segments[j], grid, work):
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
    return adjacency


def _images(first: _Segment, second: _Segment, work: _Work) -> Iterator[Image]:
    # Certified rational boxes are deliberately conservative. Near an integer
    # boundary they may admit an extra image, which exact predicates resolve.
    first_box = work.box(first.fa, first.fb)
    second_box = work.box(second.fa, second.fb)
    bounds = tuple(
        (
            ceil(first_box[k][0] - second_box[k][1]),
            floor(first_box[k][1] - second_box[k][0]),
        )
        for k in range(2)
    )
    yield from product(
        range(bounds[0][0], bounds[0][1] + 1),
        range(bounds[1][0], bounds[1][1] + 1),
    )


def _interior(a: Point, b: Point, p: Point, work: _Work) -> bool:
    return (
        p != a
        and p != b
        and all(min(a[k], b[k]) <= p[k] <= max(a[k], b[k]) for k in range(2))
        and work.orient(a, b, p) == 0
    )


def _conflict(first: _Segment, second: _Segment, grid: _Grid, work: _Work) -> bool:
    work.spend("edge_conflict_queries")
    for image in _images(first, second, work):
        work.spend("periodic_edge_image_pairs")
        c, d = _lift(second.a, image, grid.vectors), _lift(second.b, image, grid.vectors)
        a, b = first.a, first.b
        if any(
            max(a[k], b[k]) < min(c[k], d[k]) or max(c[k], d[k]) < min(a[k], b[k])
            for k in range(2)
        ):
            continue
        ab_c, ab_d = work.orient(a, b, c), work.orient(a, b, d)
        cd_a, cd_b = work.orient(c, d, a), work.orient(c, d, b)
        if ab_c * ab_d < 0 and cd_a * cd_b < 0:
            return True
        if ab_c == ab_d == cd_a == cd_b == 0:
            axis = 0 if a[0] != b[0] else 1
            if max(min(a[axis], b[axis]), min(c[axis], d[axis])) < min(
                max(a[axis], b[axis]), max(c[axis], d[axis])
            ):
                return True
        if (
            _interior(a, b, c, work)
            or _interior(a, b, d, work)
            or _interior(c, d, a, work)
            or _interior(c, d, b, work)
        ):
            return True
    return False


def _compatible_edges(
    edges: list[Edge],
    segments: list[_Segment],
    penalties: list[tuple[int, int, RadicalElement]],
    grid: _Grid,
    work: _Work,
    *,
    short_quota: int | None = None,
    long_quota: int | None = None,
) -> tuple[Edge, ...]:
    """Keep maximum-cardinality completions with field-compared feasible length tails."""
    # The result owns references only; all coefficient owners belong to the
    # caller's candidate bank. Search state and adjacency die before returning.
    work.budget.reserve(0, 64 + 16 * len(edges))
    with work.budget.temporary_scope(), work.box_scope():
        work.budget.reserve(
            0,
            2048
            + 16 * _bitset_bytes(len(edges))
            + sum(_scalar_bytes(penalty[2]) for penalty in penalties),
        )
        return _compatible_edges_search(
            edges,
            segments,
            penalties,
            grid,
            work,
            short_quota=short_quota,
            long_quota=long_quota,
        )


def _compatible_edges_search(
    edges: list[Edge],
    segments: list[_Segment],
    penalties: list[tuple[int, int, RadicalElement]],
    grid: _Grid,
    work: _Work,
    *,
    short_quota: int | None,
    long_quota: int | None,
) -> tuple[Edge, ...]:
    short_limit = len(edges) if short_quota is None else short_quota
    long_limit = len(edges) if long_quota is None else long_quota
    adjacency = _conflict_graph(segments, grid, work)
    work.budget.reserve(0, 1024 + 4 * _bitset_bytes(len(edges)))
    # A frontier retains the least geometric cost for every attainable pair of
    # tail counts at the maximum cardinality. Components combine additively;
    # throwing away a locally longer tail before this combination loses valid
    # global quantile completions.
    cache: dict[
        int,
        tuple[int, dict[tuple[int, int], tuple[RadicalElement, int]]],
    ] = {0: (0, {(0, 0): (RadicalElement(), 0)})}

    def retain(
        frontier: dict[tuple[int, int], tuple[RadicalElement, int]],
        tails: tuple[int, int],
        cost: RadicalElement,
        selected: int,
    ) -> None:
        work.spend("completion_tail_states")
        if tails[0] > short_limit or tails[1] > long_limit:
            return
        old = frontier.get(tails)
        if old is None:
            work.budget.reserve(
                0, 512 + _bitset_bytes(len(edges)) + _scalar_bytes(cost)
            )
            frontier[tails] = cost, selected
        elif cost < old[0]:
            work.budget.reserve(0, max(0, _scalar_bytes(cost) - _scalar_bytes(old[0])))
            frontier[tails] = cost, selected

    def independent(
        mask: int,
    ) -> tuple[int, dict[tuple[int, int], tuple[RadicalElement, int]]]:
        old = cache.get(mask)
        if old is not None:
            return old
        work.spend("independent_graph_states")
        work.spend("independent_graph_vertex_visits", 2 * mask.bit_count())
        work.budget.reserve(0, 1024 + 12 * _bitset_bytes(len(edges)))
        isolates, best, degree = 0, -1, -1
        scan = mask
        while scan:
            bit = scan & -scan
            vertex = bit.bit_length() - 1
            current = (adjacency[vertex] & mask).bit_count()
            if current == 0:
                isolates |= bit
            elif current > degree:
                best, degree = vertex, current
            scan ^= bit
        frontier: dict[tuple[int, int], tuple[RadicalElement, int]] = {}
        if isolates:
            size, child = independent(mask ^ isolates)
            short_count, long_count, square = 0, 0, RadicalElement()
            scan = isolates
            while scan:
                bit = scan & -scan
                below, above, length = penalties[bit.bit_length() - 1]
                short_count += below
                long_count += above
                square += length
                scan ^= bit
            for tails, (cost, selected) in child.items():
                retain(
                    frontier,
                    (tails[0] + short_count, tails[1] + long_count),
                    cost + square,
                    selected | isolates,
                )
            result = size + isolates.bit_count(), frontier
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
                result = left_size + right_size, frontier
            else:
                bit = 1 << best
                included_size, included = independent(mask & ~bit & ~adjacency[best])
                excluded_size, excluded = independent(mask & ~bit)
                size = max(included_size + 1, excluded_size)
                if included_size + 1 == size:
                    below, above, length = penalties[best]
                    for tails, (cost, selected) in included.items():
                        retain(
                            frontier,
                            (tails[0] + below, tails[1] + above),
                            cost + length,
                            selected | bit,
                        )
                if excluded_size == size:
                    for tails, (cost, selected) in excluded.items():
                        retain(frontier, tails, cost, selected)
                result = size, frontier
        cache[mask] = result
        return result

    try:
        _, frontier = independent((1 << len(edges)) - 1)
        if not frontier:
            raise ValueError(
                "periodic.algebraic.completion must satisfy requested quantile tail bounds; "
                f"got no maximum-cardinality completion with {short_limit=}, {long_limit=}"
            )
        _, selected = min(frontier.values())
        return tuple(edge for i, edge in enumerate(edges) if selected & (1 << i))
    finally:
        cache.clear()
        adjacency.clear()
        # The recursive closure otherwise keeps its coefficient/cache banks
        # alive until cyclic GC, beyond the admitted search lifetime.
        del independent


def _target_graph(
    points: tuple[Point, ...], grid: _Grid, work: _Work, quantiles: tuple[float, ...]
) -> tuple[Edge, ...]:
    # A compatible straight-line graph on this torus has at most 3V edges.
    # Its returned keys survive the candidate/search workspace.
    work.budget.reserve(0, 512 + 768 * len(points))
    with (
        work.budget.temporary_scope(),
        work.box_scope(),
        work.budget.live_storage() as candidates,
        work.budget.live_storage() as selected,
    ):
        field_bytes = max((_point_bytes(point) for point in points), default=64)
        field_bytes += sum(_point_bytes(vector) for vector in grid.vectors)
        # One field-operation workspace, plus fractional coordinates, singleton
        # records and fixed segments. No all-pairs allocation is made.
        work.budget.reserve(
            0,
            128 * field_bytes
            + sum(8 * _point_bytes(point) + 256 for point in points)
            + len(grid.protected) * 16 * field_bytes,
        )
        return _target_graph_search(points, grid, work, quantiles, candidates, selected)


def _target_graph_search(
    points: tuple[Point, ...],
    grid: _Grid,
    work: _Work,
    quantiles: tuple[float, ...],
    candidates: CoordinateLiveStorage,
    selected: CoordinateLiveStorage,
) -> tuple[Edge, ...]:
    protected = {edge: _segment(edge, points, grid) for edge in grid.protected}
    fractional = tuple(_fractional(point, grid) for point in points)
    bounds = tuple(grid.maximum * (abs(row[0]) + abs(row[1])) for row in grid.dual)
    singletons = [
        _Segment(point, point, fp, fp)
        for point, fp in zip(points, fractional, strict=True)
    ]
    singleton_index = _PeriodicBins(singletons, work)
    protected_edges = list(protected)
    protected_segments = list(protected.values())
    protected_index = _PeriodicBins(protected_segments, work)
    radius2 = grid.radius * grid.radius
    maximum2 = grid.maximum * grid.maximum
    candidates.set_bound(256)
    bank: list[tuple[Edge, _Segment, RadicalElement]] = []
    for a in range(len(points)):
        for b in range(a + 1, len(points)):
            work.spend("root_orbit_pairs")
            ranges = tuple(
                range(
                    ceil(fractional[a][k] - fractional[b][k] - bounds[k]),
                    floor(fractional[a][k] - fractional[b][k] + bounds[k]) + 1,
                )
                for k in range(2)
            )
            for sx, sy in product(ranges[0], ranges[1]):
                work.spend("bounded_edge_candidates")
                edge = (a, b, sx, sy)
                stop = _lift(points[b], (sx, sy), grid.vectors)
                square = _square(points[a], stop)
                if square <= maximum2:
                    segment = _segment(edge, points, grid)
                    # Arithmetic is covered by the field workspace; admit the
                    # surviving coefficient owners before retaining this entry.
                    candidates.set_bound(
                        candidates.bound + 384
                        + _segment_bytes(segment) + _scalar_bytes(square)
                    )
                    bank.append((edge, segment, square))

    def visible(edge: Edge, segment: _Segment) -> bool:
        for root in singleton_index.candidates(segment):
            singleton = singletons[root]
            if root in edge[:2]:
                continue
            work.spend("root_edge_visibility_pairs")
            if any(
                _interior(
                    segment.a,
                    segment.b,
                    _lift(singleton.a, image, grid.vectors),
                    work,
                )
                for image in _images(segment, singleton, work)
            ):
                return False
        return True

    radius_edges: list[Edge] = []
    radius_segments: list[_Segment] = []
    for edge, segment, square in bank:
        if square != radius2:
            continue
        if any(
            _conflict(segment, protected_segments[index], grid, work)
            for index in protected_index.candidates(segment)
            if protected_edges[index] != edge
        ) or not visible(edge, segment):
            continue
        work.budget.reserve(0, 192)
        radius_edges.append(edge)
        radius_segments.append(segment)
    work.budget.reserve(0, 256 + 256 * len(radius_edges))
    selected_radius = _compatible_edges(
        radius_edges,
        radius_segments,
        [(0, 0, RadicalElement()) for _ in radius_edges],
        grid,
        work,
    )
    work.budget.reserve(0, 512 + 256 * (len(protected) + len(selected_radius)))
    mandatory = dict(protected)
    radius_lookup = dict(zip(radius_edges, radius_segments, strict=True))
    mandatory.update((edge, radius_lookup[edge]) for edge in selected_radius)
    mandatory_segments = list(mandatory.values())
    mandatory_index = _PeriodicBins(mandatory_segments, work)
    selected.set_bound(
        1024 + sum(256 + _segment_bytes(segment) for segment in mandatory_segments)
    )
    radius_lookup.clear()
    radius_edges.clear()
    radius_segments.clear()
    optional: list[Edge] = []
    optional_segments: list[_Segment] = []
    optional_penalties: list[tuple[int, int, RadicalElement]] = []
    for edge, segment, square in bank:
        if edge in mandatory:
            continue
        if any(
            _conflict(segment, mandatory_segments[index], grid, work)
            for index in mandatory_index.candidates(segment)
        ):
            continue
        if not visible(edge, segment):
            continue
        selected.set_bound(
            selected.bound + 384 + _segment_bytes(segment) + _scalar_bytes(square)
        )
        optional.append(edge)
        optional_segments.append(segment)
        optional_penalties.append((int(square < radius2), int(square > radius2), square))
    # The construction supplies an algebraic target plateau. For each requested
    # interpolation, both rank endpoints must belong to that plateau. These
    # integer tail bounds come from the declared requested quantile ranks,
    # not a source-specific edge-count target.
    edge_count = 3 * len(points)
    ranks = tuple(quantile * (edge_count - 1) for quantile in quantiles)
    short_quota = min(floor(rank) for rank in ranks)
    long_quota = edge_count - 1 - max(ceil(rank) for rank in ranks)
    # Only mandatory classifications consume lengths now; do not duplicate the
    # entire candidate bank in a square-length dictionary.
    work.budget.reserve(0, 256 + 128 * len(mandatory))
    mandatory_squares = {edge: square for edge, _, square in bank if edge in mandatory}
    for edge in mandatory:
        work.spend("mandatory_edge_classifications")
        square = mandatory_squares[edge]
        short_quota -= int(square < radius2)
        long_quota -= int(square > radius2)
    mandatory_squares.clear()
    bank.clear()
    candidates.set_bound(0)
    work.discard_boxes_except(optional_segments)
    completion = _compatible_edges(
        optional,
        optional_segments,
        optional_penalties,
        grid,
        work,
        short_quota=short_quota,
        long_quota=long_quota,
    )
    return tuple(sorted(set(mandatory) | set(completion)))


def _faces(
    edges: tuple[Edge, ...], points: tuple[Point, ...], grid: _Grid, work: _Work
) -> list[list[Corner]]:
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
        a, b = directions[i], directions[j]
        ah = a[1] > 0 or (a[1] == 0 and a[0] >= 0)
        bh = b[1] > 0 or (b[1] == 0 and b[0] >= 0)
        if ah != bh:
            return -1 if ah else 1
        cross = work.orient((RadicalElement(), RadicalElement()), a, b)
        if not cross:
            raise ValueError(
                "periodic.algebraic.rays must have distinct directions; got coincident source graph rays"
            )
        return -cross

    positions: dict[int, int] = {}
    for root, star in enumerate(outgoing):
        if len(star) < 2:
            raise ValueError(
                f"periodic.algebraic.completed_graph must include every constructed root; got root={root}, degree={len(star)}, edges={len(edges)}"
            )
        star.sort(key=cmp_to_key(compare))
        positions.update((edge, i) for i, edge in enumerate(star))
    visited: set[int] = set()
    faces: list[list[Corner]] = []
    for start in range(len(half)):
        if start in visited:
            continue
        current, image = start, (0, 0)
        face: list[Corner] = []
        while current not in visited:
            work.spend("face_walk_steps")
            visited.add(current)
            a, b, sx, sy = half[current]
            face.append((a, image))
            image = (image[0] + sx, image[1] + sy)
            current = outgoing[b][(positions[current ^ 1] - 1) % len(outgoing[b])]
        if current != start or image != (0, 0) or len(face) < 3:
            raise ValueError(
                "periodic.algebraic.faces must be contractible source polygons; got an open or winding face"
            )
        faces.append(face)
    return faces


def _bind_materials(
    triangles: list[tuple[Corner, Corner, Corner]],
    points: tuple[Point, ...],
    grid: _Grid,
    work: _Work,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    source_triangles = [
        tuple(
            _lift(grid.points[int(root)], (int(image[0]), int(image[1])), grid.vectors)
            for root, image in zip(cell, images, strict=True)
        )
        for cell, images in zip(grid.strata.cells, grid.strata.shifts, strict=True)
    ]
    source_fractional = [
        tuple(_fractional(point, grid) for point in triangle)
        for triangle in source_triangles
    ]
    regions: list[int] = []
    ancestry: list[tuple[int, ...]] = []
    for triangle in triangles:
        corners = tuple(
            _lift(points[root], image, grid.vectors) for root, image in triangle
        )
        fractional = tuple(_fractional(point, grid) for point in corners)
        members: set[int] = set()
        for parent, (source, sf) in enumerate(
            zip(source_triangles, source_fractional, strict=True)
        ):
            work.spend("source_cell_pairs")
            ranges = tuple(
                range(
                    ceil(min(p[k] for p in fractional) - max(p[k] for p in sf)),
                    floor(max(p[k] for p in fractional) - min(p[k] for p in sf)) + 1,
                )
                for k in range(2)
            )
            for sx, sy in product(ranges[0], ranges[1]):
                work.spend("source_overlap_image_pairs")
                other = tuple(_lift(point, (sx, sy), grid.vectors) for point in source)
                separated = False
                for first, second in ((corners, other), (other, corners)):
                    if any(
                        all(
                            work.orient(first[k], first[(k + 1) % 3], p) <= 0
                            for p in second
                        )
                        for k in range(3)
                    ):
                        separated = True
                        break
                if not separated:
                    members.add(parent)
        labels = {int(grid.strata.regions[parent]) for parent in members}
        if len(labels) != 1:
            raise ValueError(
                "periodic.algebraic.material must resolve one source region per positive cell; got absent or crossed source strata"
            )
        regions.append(next(iter(labels)))
        ancestry.append(tuple(sorted(members)))
    return np.asarray(regions, dtype=np.int64), tuple(ancestry)


def _source_region_areas(grid: _Grid) -> dict[int, RadicalElement]:
    expected: dict[int, RadicalElement] = {}
    for cell, images, region in zip(
        grid.strata.cells, grid.strata.shifts, grid.strata.regions, strict=True
    ):
        corners = tuple(
            _lift(grid.points[int(root)], (int(image[0]), int(image[1])), grid.vectors)
            for root, image in zip(cell, images, strict=True)
        )
        delta, other = _sub(corners[1], corners[0]), _sub(corners[2], corners[0])
        label = int(region)
        expected[label] = (
            expected.get(label, RadicalElement())
            + (delta[0] * other[1] - delta[1] * other[0]) / 2
        )
    return expected


def _certify_carrier(
    triangles: list[tuple[Corner, Corner, Corner]],
    regions: np.ndarray,
    points: tuple[Point, ...],
    vectors: tuple[Point, Point],
    protected: tuple[Edge, ...],
    expected: dict[int, RadicalElement],
    work: _Work,
) -> None:
    """Prove positive cells, closed incidence, manifold links, and region areas."""
    incidence: dict[Edge, list[int]] = {}
    links: dict[int, list[tuple[Corner, Corner]]] = {}
    areas: dict[int, RadicalElement] = {}
    for triangle, region in zip(triangles, regions, strict=True):
        corners = tuple(_lift(points[root], image, vectors) for root, image in triangle)
        if work.orient(corners[0], corners[1], corners[2]) <= 0:
            raise ValueError(
                "periodic.algebraic.orientation must be strictly positive; got a nonpositive completion cell"
            )
        delta, other = _sub(corners[1], corners[0]), _sub(corners[2], corners[0])
        area = (delta[0] * other[1] - delta[1] * other[0]) / 2
        label = int(region)
        areas[label] = areas.get(label, RadicalElement()) + area
        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            root, image = triangle[a]
            first, fi = triangle[b]
            second, si = triangle[c]
            edge, sign = _key(root, image, first, fi)
            incidence.setdefault(edge, []).append(sign)
            links.setdefault(root, []).append(
                (
                    (first, (fi[0] - image[0], fi[1] - image[1])),
                    (second, (si[0] - image[0], si[1] - image[1])),
                )
            )
    if (
        len(incidence) != 3 * len(points)
        or any(sorted(signs) != [-1, 1] for signs in incidence.values())
        or not set(protected) <= incidence.keys()
    ):
        raise ValueError(
            "periodic.algebraic.incidence must be closed and retain protected windings; got an invalid edge bank"
        )
    for pairs in links.values():
        adjacency: dict[Corner, list[Corner]] = {}
        for a, b in pairs:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        if any(len(neighbors) != 2 for neighbors in adjacency.values()):
            raise ValueError(
                "periodic.algebraic.vertex_link must be 2-regular; got a nonmanifold source-action vertex"
            )
        pending, seen = [min(adjacency)], set()
        while pending:
            work.spend("vertex_link_steps")
            corner = pending.pop()
            if corner not in seen:
                seen.add(corner)
                pending.extend(adjacency[corner])
        if len(seen) != len(adjacency):
            raise ValueError(
                "periodic.algebraic.vertex_link must be connected; got multiple source-action link cycles"
            )
    if areas != expected or len(links) != len(points):
        raise ValueError(
            "periodic.algebraic.material_areas must equal the source partition and include every site; got an incomplete carrier"
        )


@checked
def construct_exact_algebraic_carrier(
    source: source_owner.NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> ExactAlgebraicCarrier:
    """Construct the admitted algebraic source-action carrier.

    Admission proved commuting primitive source actions, two opposite protected
    material cycles, a rational lattice metric, and the small positive four-band
    branch; execution renews that proof on its own ledger and refuses a changed
    premise. The finite radius-edge and bounded-completion independent-set
    programs use exact predicates; neither topology nor site coordinates are persisted in
    this construction module. Every operation is charged to the request ticket.
    """
    limits = specification.limits
    starting_work = budget.work_units
    with (
        _phase(scope, budget, limits.maximum_work_units, limits.maximum_scratch_bytes),
        budget.temporary_scope(),
    ):
        work = _Work(budget)
        grid = _recover_grid(source, specification, work)
        if grid is None:
            raise MeshingFailure(
                MeshingFailureCategory.AUDIT_FAILED,
                "periodic.algebraic.source_actions must retain the admitted construction premises; got a changed source-action configuration",
            )
        # This global reconstruction replaces the authored source-cell set.
        # Admit that touched-cell cavity through the native shared ledger before
        # allocating the generated topology; a refusal retains native evidence.
        cavity_cells = grid.strata.cells.shape[0]
        scope.admit_host_cavity(cavity_cells)
        carrier_count = 4 * len(grid.points) + 3 * len(grid.rows)
        # Source coordinates/identities survive graph construction. Admit a
        # linear site bank and one field workspace, not a nonexistent V² bank.
        field_bytes = (
            max(_point_bytes(point) for point in grid.points)
            + sum(_point_bytes(vector) for vector in grid.vectors)
            + _scalar_bytes(grid.run)
            + _scalar_bytes(grid.height)
        )
        budget.reserve(0, carrier_count * (512 + 8 * field_bytes) + 128 * field_bytes)
        coordinates, identities = _sites(grid, work)
        count = len(coordinates)
        if count > limits.maximum_vertices or 2 * count > limits.maximum_cells:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "periodic.algebraic.entities must fit the request vertex and cell bounds; got a larger source-action carrier",
            )
        quantiles = tuple(
            size_statistic_quantile(statistic)
            for statistic in specification.size_compliance.target_statistics
        )
        edges = _target_graph(coordinates, grid, work, quantiles)
        # The final topology, its corners, source-overlap preparation and the
        # manifold witness coexist. Their cardinalities are linear in the
        # actual edge/cell banks, independent of rejected candidate pairs.
        budget.reserve(
            0,
            (2 * count + cavity_cells) * (2048 + 8 * field_bytes)
            + 512 * len(edges),
        )
        faces = _faces(edges, coordinates, grid, work)
        if any(len(face) != 3 for face in faces):
            raise ValueError(
                "periodic.algebraic.completed_graph must triangulate the source torus; "
                f"got face_sizes={tuple(len(face) for face in faces)}, edges={len(edges)}"
            )
        triangles = [(face[0], face[1], face[2]) for face in faces]
        faces.clear()
        if len(triangles) != 2 * count:
            raise ValueError(
                "periodic.algebraic.cells must close the source torus; got an inconsistent Euler count"
            )
        regions, ancestry = _bind_materials(triangles, coordinates, grid, work)
        _certify_carrier(
            triangles,
            regions,
            coordinates,
            grid.vectors,
            grid.protected,
            _source_region_areas(grid),
            work,
        )
        work.spend("representative_coordinates", 2 * count)
        points = np.asarray(
            [
                [value.rounded(budget=budget)[0] for value in point]
                for point in coordinates
            ],
            dtype=np.float64,
        )
        simplices = np.asarray(
            [[root for root, _ in triangle] for triangle in triangles], dtype=np.int64
        )
        shifts = np.asarray(
            [[image for _, image in triangle] for triangle in triangles], dtype=np.int64
        )
        for array in (points, simplices, shifts, regions):
            array.setflags(write=False)
        identity = canonical_fingerprint(
            {
                "kind": "source-action-four-band-algebraic-carrier",
                "source_binding": source.binding_id,
                "size_compliance": specification.size_compliance.policy_id,
                "sites": identities,
                "topology_rule": "maximum-compatible-radius-edges-quantile-feasible-cardinality-minimum-square-completion",
            }
        )
        carrier = ExactAlgebraicCarrier(
            points=points,
            exact_coordinates=coordinates,
            simplices=simplices,
            shifts=shifts,
            regions=regions,
            ancestry=ancestry,
            work=budget.work_units - starting_work,
            image_count=0,
            source_identity=identity,
            site_identities=identities,
            work_evidence=tuple(sorted(work.counts.items())),
        )
        budget.retain_basis((carrier,))
        return carrier


@contextmanager
def _admission_budget(limits: MeshingLimits) -> Iterator[CoordinateEnclosureBudget]:
    """Bound an admission proof by the request's own work and storage limits.

    An exhausted proof is a resource refusal of the request, never evidence that
    its premises fail; callers must not select another construction from it.
    """
    ticket = HostBatchTicket(
        limits.maximum_work_units,
        queries_upper=limits.maximum_geometry_queries,
        bytes_upper=limits.maximum_scratch_bytes,
    )
    budget = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=limits.maximum_work_units,
        maximum_memory_bytes=limits.maximum_scratch_bytes,
    )
    try:
        with ticket, budget.activate():
            yield budget
    except CoordinateEnclosureResourceError as error:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "periodic.algebraic.admission must fit the request work and storage allowances; got a resource refusal",
            requested=(
                (str(error.resource), float(error.requested)),
                ("limit", float(error.limit)),
            ),
            achieved=(("completed", float(error.completed)),),
        ) from error


@checked
def prove_source_action_bands(
    source: source_owner.NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
) -> bool:
    """Admit the four-band carrier exactly when its source-action premises hold."""
    with _admission_budget(specification.limits) as budget:
        return _recover_grid(source, specification, _Work(budget)) is not None


type _Vector = tuple[Fraction, Fraction]


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


def _hard_uniform_controls(
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
) -> list[UniformSizeControl] | None:
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
    return m * a[0] + k * b[0], m * a[1] + k * b[1]


def _rational_fractional(point: _Vector, dual: tuple[_Vector, _Vector]) -> _Vector:
    return _dot(dual[0], point), _dot(dual[1], point)


def _integer_lattice_basis(
    generators: tuple[tuple[int, int], ...], work: _Work
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the Hermite basis ``((alpha, 0), (beta, gamma))`` of a full-rank span."""
    pivot_x = pivot_y = alpha = 0
    for x, y in generators:
        work.spend("lattice_basis_generators")
        while y:
            work.spend("lattice_basis_steps")
            quotient = pivot_y // y
            pivot_x, pivot_y, x, y = x, y, pivot_x - quotient * x, pivot_y - quotient * y
        alpha = gcd(alpha, x)
    if pivot_y < 0:
        pivot_x, pivot_y = -pivot_x, -pivot_y
    return (alpha, 0), (pivot_x % alpha, pivot_y)


def _reduced_basis(
    first: _Vector, second: _Vector, work: _Work
) -> tuple[_Vector, _Vector]:
    """Lagrange-reduce a rational lattice basis and orient it positively."""
    while True:
        work.spend("lattice_reduction_steps")
        if _dot(first, first) > _dot(second, second):
            first, second = second, first
        quotient = round(_dot(first, second) / _dot(first, first))
        if not quotient:
            break
        second = _combine(1, second, -quotient, first)
    if _cross(first, second) < 0:
        second = (-second[0], -second[1])
    return first, second


def _bezout(a: int, b: int) -> tuple[int, int]:
    old_r, r, old_s, s, old_t, t = a, b, 1, 0, 0, 1
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    return (-old_s, -old_t) if old_r < 0 else (old_s, old_t)


def _target_steps(
    first: _Vector, second: _Vector, target: Fraction, work: _Work
) -> list[tuple[_Vector, _Vector, int]]:
    """Enumerate primitive lattice vectors of exact target length with complements.

    The reduced Gram form bounds both coefficients, so every vector of length
    ``target`` is visited. Each step keeps one sign; its complement is normalized
    to ``0 <= step.rise < target^2`` and ``lead`` selects the next-row seed whose
    projection follows the apex ``step / 2``.
    """
    square = target * target
    g11, g12, g22 = _dot(first, first), _dot(first, second), _dot(second, second)
    bound = isqrt(floor(square * g11 / (g11 * g22 - g12 * g12)))
    steps: list[tuple[_Vector, _Vector, int]] = []
    for k in range(-bound, bound + 1):
        discriminant = (g12 * k) ** 2 - g11 * (g22 * k * k - square)
        if discriminant < 0:
            continue
        radius = isqrt(floor(discriminant / (g11 * g11))) + 1
        center = -g12 * k / g11
        for m in range(floor(center) - radius, ceil(center) + radius + 1):
            work.spend("row_step_candidates")
            if gcd(m, k) != 1 or g11 * m * m + 2 * g12 * m * k + g22 * k * k != square:
                continue
            step = _combine(m, first, k, second)
            if step[1] < 0 or (step[1] == 0 and step[0] < 0):
                continue
            # m*x + k*y = 1 makes (step, rise) unimodular with the basis orientation.
            x, y = _bezout(m, k)
            rise = _combine(-y, first, x, second)
            rise = _combine(1, rise, -floor(_dot(step, rise) / square), step)
            steps.append((step, rise, int(2 * _dot(step, rise) < square)))
    return sorted(steps, key=lambda item: (item[0][1], item[0][0]))


def _band_apex(step: _Vector, budget: CoordinateEnclosureBudget) -> Point:
    height = RadicalElement.sqrt(Fraction(3, 4), budget=budget)
    return (
        RadicalElement(step[0] / 2) - height * step[1],
        RadicalElement(step[1] / 2) + height * step[0],
    )


def _orbit_root(proof: OrbitLatticeBands, point: _Vector, work: _Work) -> Corner:
    """Name the seed orbit and lattice image of a point of the admitted coset."""
    work.spend("orbit_root_lookups")
    fractional = _rational_fractional(point, proof.dual)
    root = proof.residues[
        (fractional[0] - floor(fractional[0]), fractional[1] - floor(fractional[1]))
    ]
    base = proof.fractional[root]
    image = (fractional[0] - base[0], fractional[1] - base[1])
    if image[0].denominator != 1 or image[1].denominator != 1:
        raise ValueError(
            "periodic.algebraic.orbit_lattice.image must be an integer lattice action; got a point outside the admitted coset"
        )
    return root, (int(image[0]), int(image[1]))


def _band_neighbors(
    proof: OrbitLatticeBands, root: int, work: _Work
) -> tuple[Corner, Corner, Corner, Corner]:
    """Return the next, previous, leading and trailing next-row seed corners."""
    point, step, rise = proof.roots[root], proof.step, proof.rise
    return (
        _orbit_root(proof, _combine(1, point, 1, step), work),
        _orbit_root(proof, _combine(1, point, -1, step), work),
        _orbit_root(
            proof, _combine(1, _combine(1, point, proof.lead, step), 1, rise), work
        ),
        _orbit_root(
            proof, _combine(1, _combine(1, point, proof.lead - 1, step), 1, rise), work
        ),
    )


def _band_size_issues(
    proof: OrbitLatticeBands,
    controls: list[UniformSizeControl],
    policy: SizeCompliancePolicy,
    work: _Work,
) -> bool:
    """Run the exact hard verdict on the closed-form edge bank before construction.

    Per root the carrier has one seed-row, one apex-row and two equilateral edges
    of target length, and one leading and one trailing filler edge whose squares
    are field elements. Their quotient endpoints are the admitted seed orbits.
    """
    count = len(proof.roots)
    apex = _band_apex(proof.step, work.budget)

    def filler(shift: int) -> RadicalElement:
        offset = _combine(shift, proof.step, 1, proof.rise)
        x, y = offset[0] - apex[0], offset[1] - apex[1]
        return x * x + y * y

    target, leading, trailing = (
        RadicalElement(proof.target * proof.target),
        filler(proof.lead),
        filler(proof.lead - 1),
    )
    work.budget.reserve(0, 384 * count)
    squared = np.empty(6 * count, dtype=object)
    pairs = np.empty((6 * count, 2), dtype=np.int64)
    for root in range(count):
        right, _, ahead, behind = _band_neighbors(proof, root, work)
        top = count + root
        for offset, (first, second, value) in enumerate(
            (
                (root, right[0], target),
                (top, count + right[0], target),
                (root, top, target),
                (right[0], top, target),
                (top, ahead[0], leading),
                (top, behind[0], trailing),
            )
        ):
            squared[6 * root + offset] = value
            pairs[6 * root + offset] = first, second
    return any(
        periodic_owner.periodic_size_control_issues(
            squared, pairs, control, policy, budget=work.budget
        )
        for control in controls
    )


@checked
def prove_orbit_lattice_bands(
    source: source_owner.NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> OrbitLatticeBands | None:
    """Admit equilateral row bands only from exact seed and request premises.

    The premises are a zero-tolerance hard uniform target, seed representatives
    exhausting one coset of a translation lattice, equilateral apexes strictly
    between consecutive seed rows, and a passing exact hard verdict of the
    closed-form edge bank. ``None`` means a premise does not hold; resource
    refusals propagate and never select another construction.
    """
    if budget is None:
        with _admission_budget(specification.limits) as owned:
            return prove_orbit_lattice_bands(source, specification, budget=owned)
    match source.domain:
        case periodic_owner.PeriodicPointOrbits() as orbits:
            pass
        case CellMesh():
            return None
        case _ as invalid:
            assert_never(invalid)
    policy = specification.size_compliance
    controls = _hard_uniform_controls(specification)
    if (
        source.ambient_dimension != 2
        or policy.absolute_tolerance != 0
        or policy.relative_tolerance != 0
        or not controls
    ):
        return None
    target = Fraction(float(controls[0].target_size))
    if any(Fraction(float(control.target_size)) != target for control in controls):
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
    representatives = np.asarray(orbits.representatives, dtype=np.float64)
    count = representatives.shape[0]
    work = _Work(budget)
    work.spend("orbit_lattice_roots", count)
    budget.reserve(0, 1024 * count)
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
        if not _band_size_issues(proof, controls, policy, work):
            return proof
    return None


@checked
def construct_orbit_lattice_band_carrier(
    source: source_owner.NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> ExactAlgebraicCarrier:
    """Construct the admitted orbit-lattice equilateral band carrier.

    Execution renews the admission proof on its own ledger and refuses a changed
    premise. The authored seed roots stay first and unchanged; one apex per root
    closes an equilateral row band and the ordered merge of an apex row with the
    next seed row closes each filler band. Every cell, incidence, vertex link and
    the exact quotient area are certified, and the hard verdict is rerun on the
    constructed field-valued edges.
    """
    limits = specification.limits
    starting_work = budget.work_units
    with (
        _phase(scope, budget, limits.maximum_work_units, limits.maximum_scratch_bytes),
        budget.temporary_scope(),
    ):
        proof = prove_orbit_lattice_bands(source, specification, budget=budget)
        controls = _hard_uniform_controls(specification)
        if proof is None or not controls:
            raise MeshingFailure(
                MeshingFailureCategory.AUDIT_FAILED,
                "periodic.algebraic.orbit_lattice must retain the admitted construction premises; got a changed seed-lattice configuration",
            )
        roots = len(proof.roots)
        count = 2 * roots
        if count > limits.maximum_vertices or 2 * count > limits.maximum_cells:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "periodic.algebraic.entities must fit the request vertex and cell bounds; got a larger orbit-lattice carrier",
            )
        work = _Work(budget)
        budget.reserve(0, 4096 * count)
        a, b = proof.vectors
        vectors = (
            (RadicalElement(a[0]), RadicalElement(a[1])),
            (RadicalElement(b[0]), RadicalElement(b[1])),
        )
        apex = _band_apex(proof.step, budget)
        coordinates: list[Point] = [
            (RadicalElement(x), RadicalElement(y)) for x, y in proof.roots
        ]
        identities: list[SiteIdentity] = [
            ("authored_source_root", (root,)) for root in range(roots)
        ]
        wraps: list[Image] = []
        for root in range(roots):
            work.spend("band_apex_sites")
            raw = _add(coordinates[root], apex)
            fractional = _fractional(raw, proof)
            image = (floor(fractional[0]), floor(fractional[1]))
            coordinates.append(_lift(raw, (-image[0], -image[1]), vectors))
            wraps.append(image)
            identities.append(("orbit_lattice_band_apex", (root,)))

        def apex_corner(corner: Corner) -> Corner:
            root, image = corner
            return roots + root, (image[0] + wraps[root][0], image[1] + wraps[root][1])

        triangles: list[tuple[Corner, Corner, Corner]] = []
        for root in range(roots):
            work.spend("band_cells", 4)
            right, left, ahead, behind = _band_neighbors(proof, root, work)
            here: Corner = (root, (0, 0))
            top = apex_corner(here)
            triangles.extend(
                (
                    (here, right, top),
                    (here, top, apex_corner(left)),
                    (top, apex_corner(right), ahead),
                    (top, ahead, behind),
                )
            )
        points = tuple(coordinates)
        regions = np.zeros(len(triangles), dtype=np.int64)
        _certify_carrier(
            triangles,
            regions,
            points,
            vectors,
            (),
            {0: RadicalElement(_cross(a, b))},
            work,
        )
        squares: dict[Edge, RadicalElement] = {}
        for triangle in triangles:
            for (first, fi), (second, si) in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ):
                edge, _ = _key(first, fi, second, si)
                if edge not in squares:
                    work.spend("carrier_edge_squares")
                    squares[edge] = _square(
                        points[edge[0]],
                        _lift(points[edge[1]], (edge[2], edge[3]), vectors),
                    )
        squared = np.empty(len(squares), dtype=object)
        for index, value in enumerate(squares.values()):
            squared[index] = value
        pairs = np.asarray([edge[:2] for edge in squares], dtype=np.int64)
        if any(
            periodic_owner.periodic_size_control_issues(
                squared, pairs, control, specification.size_compliance, budget=budget
            )
            for control in controls
        ):
            raise MeshingFailure(
                MeshingFailureCategory.AUDIT_FAILED,
                "periodic.algebraic.orbit_lattice.size must reproduce the admitted exact hard verdict; got hard issues on the constructed edges",
            )
        work.spend("representative_coordinates", 2 * count)
        rounded = np.asarray(
            [[value.rounded(budget=budget)[0] for value in point] for point in points],
            dtype=np.float64,
        )
        simplices = np.asarray(
            [[root for root, _ in triangle] for triangle in triangles], dtype=np.int64
        )
        shifts = np.asarray(
            [[image for _, image in triangle] for triangle in triangles], dtype=np.int64
        )
        for array in (rounded, simplices, shifts, regions):
            array.setflags(write=False)
        identity = canonical_fingerprint(
            {
                "kind": "orbit-lattice-equilateral-band-algebraic-carrier",
                "source_binding": source.binding_id,
                "size_compliance": specification.size_compliance.policy_id,
                "hard_controls": [control.control_id for control in controls],
                "row_step": [[part.numerator, part.denominator] for part in proof.step],
                "row_rise": [[part.numerator, part.denominator] for part in proof.rise],
                "lead": proof.lead,
                "sites": identities,
                "topology_rule": "equilateral-row-bands-with-ordered-row-merge-fillers",
            }
        )
        carrier = ExactAlgebraicCarrier(
            points=rounded,
            exact_coordinates=points,
            simplices=simplices,
            shifts=shifts,
            regions=regions,
            ancestry=tuple((index,) for index in range(len(triangles))),
            work=budget.work_units - starting_work,
            image_count=0,
            source_identity=identity,
            site_identities=tuple(identities),
            work_evidence=tuple(sorted(work.counts.items())),
        )
        budget.retain_basis((carrier,))
        return carrier
