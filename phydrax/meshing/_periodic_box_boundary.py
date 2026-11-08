#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Well-shaped boundary of a periodic trigonometric region in its fundamental box.

The discovery's lifted contour (marching tetrahedra on the target-level grid)
has needles and caps wherever a root sits near a grid corner. Its cell-face
caps, tiled by grid squares cut at those roots, inherit them. A fixed-boundary
fill keeps every boundary triangle, so those triangles force bad tetrahedra.
This module rebuilds the boundary before the fill:

* the contour is remeshed isotropically toward one edge length. Interior
  vertices move on the zero set. Seam vertices (on one cell-face plane) move
  along the zero set's trace in that plane, and their upper-face copy moves
  with them as its exact lattice translate. Box-edge roots stay pinned. Every
  placed vertex carries the discovery's root certificate (`QuotientZeroSet`).
* each lower cell face is triangulated by a refined constrained Delaunay
  triangulation of the region's trace, bounded by the seam polyline and the
  region's arcs of the face square. A refinement point on a seam becomes a
  certified seam split of the contour; a point on a box edge becomes a box-edge
  sample shared by every copy of that edge. The upper face is the exact
  translate.

Orbits are canonical reduced coordinates. Every coordinate on an upper plane
is replaced by the lower one, which is exact because the anchored box has an
exact ``offset + 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .._meshcore import constrained_delaunay_2d
from ..geometry.implicit._periodic_discovery import (
    PeriodicImplicitSurface,
    QuotientZeroSet,
)


type _Plane = tuple[int, int]
type _Triangle = tuple[int, int, int]

# Isotropic remeshing thresholds relative to the target edge length.
_LONG = 4.0 / 3.0
_SHORT = 4.0 / 5.0
# Cap refinement: minimum angle (degrees) of the planar triangulation.
_CAP_MINIMUM_ANGLE = 25.0
# A collapse may lower its star's minimum angle only down to this (degrees).
_COLLAPSE_ANGLE = 20.0
# A new triangle's normal must lie within 60 degrees of every vertex gradient,
# unless the replaced triangles were already less aligned.
_ALIGNMENT = 0.5
# Smallest accepted gain (degrees) of a flip or relaxation.
_GAIN = 1.0e-3


@dataclass(frozen=True, slots=True)
class PeriodicBoxBoundary:
    """Lifted boundary of the fundamental box's region.

    ``points`` are lifted reduced coordinates with ``orbits`` (canonical
    reduced-coordinate classes) and ``windings``. ``surface`` holds the outward
    contour triangles. ``caps[axis]`` is the pair of lower and upper cap
    triangulations of that axis, both counter-clockwise about ``+e_axis``; the
    upper one is the exact translate of the lower one. ``root_error`` bounds
    the reduced distance of every contour vertex from a root of the field.
    """

    points: np.ndarray
    orbits: np.ndarray
    windings: np.ndarray
    surface: np.ndarray
    caps: tuple[tuple[np.ndarray, np.ndarray], ...]
    root_error: float


def _minimum_angle(corners: np.ndarray, /) -> float:
    """Smallest angle (degrees) of one triangle; zero when degenerate."""
    smallest = math.pi
    for index in range(3):
        first = corners[(index + 1) % 3] - corners[index]
        second = corners[(index + 2) % 3] - corners[index]
        norms = float(np.linalg.norm(first) * np.linalg.norm(second))
        if norms == 0.0:
            return 0.0
        cosine = float(np.dot(first, second)) / norms
        smallest = min(smallest, math.acos(min(1.0, max(-1.0, cosine))))
    return math.degrees(smallest)


class _Patch:
    """Mutable lifted contour patch with seam pairing and certified placement."""

    def __init__(
        self,
        zero_set: QuotientZeroSet,
        domain: np.ndarray,
        points: np.ndarray,
        triangles: np.ndarray,
        error: float,
        length: float,
        /,
    ) -> None:
        self.zero_set = zero_set
        self.low = np.asarray(domain[0], dtype=np.float64)
        self.high = np.asarray(domain[1], dtype=np.float64)
        self.length = float(length)
        self.points: list[np.ndarray] = [np.array(row) for row in points]
        self.gradients: list[np.ndarray] = list(zero_set.gradients(points))
        self.errors: list[float] = [float(error)] * len(self.points)
        self.planes: list[frozenset[_Plane]] = [self._planes(row) for row in self.points]
        self.alive: list[bool] = [True] * len(self.points)
        self.at: dict[bytes, int] = {}
        for vertex, planes in enumerate(self.planes):
            if planes:
                self.at[self.points[vertex].tobytes()] = vertex
        self.triangles: list[_Triangle | None] = []
        self.edges: dict[tuple[int, int], list[int]] = {}
        self.stars: list[set[int]] = [set() for _ in self.points]
        for row in np.asarray(triangles, dtype=np.int64).tolist():
            self._add((row[0], row[1], row[2]))
        self.seams_frozen = False
        for (first, second), faces in self.edges.items():
            if len(faces) == 1 and not self.common(first, second):
                raise ValueError("A contour boundary edge must lie in a cell-face plane.")

    # ------------------------------------------------------------- topology

    def _planes(self, point: np.ndarray, /) -> frozenset[_Plane]:
        planes: set[_Plane] = set()
        for axis in range(3):
            if point[axis] == self.low[axis]:
                planes.add((axis, 0))
            elif point[axis] == self.high[axis]:
                planes.add((axis, 1))
        return frozenset(planes)

    def _add(self, triangle: _Triangle, /) -> None:
        row = len(self.triangles)
        self.triangles.append(triangle)
        for index in range(3):
            first, second = triangle[index], triangle[(index + 1) % 3]
            self.edges.setdefault((min(first, second), max(first, second)), []).append(
                row
            )
            self.stars[first].add(row)

    def _remove(self, row: int, /) -> None:
        triangle = self.triangles[row]
        assert triangle is not None
        self.triangles[row] = None
        for index in range(3):
            first, second = triangle[index], triangle[(index + 1) % 3]
            key = (min(first, second), max(first, second))
            faces = self.edges[key]
            faces.remove(row)
            if not faces:
                del self.edges[key]
            self.stars[first].discard(row)

    def vertex(
        self, point: np.ndarray, gradient: np.ndarray, error: float, /
    ) -> int:
        row = len(self.points)
        self.points.append(np.array(point, dtype=np.float64))
        self.gradients.append(np.array(gradient, dtype=np.float64))
        self.errors.append(float(error))
        planes = self._planes(self.points[row])
        self.planes.append(planes)
        self.alive.append(True)
        self.stars.append(set())
        if planes:
            self.at[self.points[row].tobytes()] = row
        return row

    def replace(self, removed: list[int], added: list[_Triangle], /) -> None:
        for row in removed:
            self._remove(row)
        for triangle in added:
            self._add(triangle)

    def common(self, first: int, second: int, /) -> frozenset[_Plane]:
        return self.planes[first] & self.planes[second]

    def faces(self, first: int, second: int, /) -> list[int]:
        return self.edges.get((min(first, second), max(first, second)), [])

    def neighbors(self, vertex: int, /) -> set[int]:
        result: set[int] = set()
        for row in self.stars[vertex]:
            triangle = self.triangles[row]
            assert triangle is not None
            result.update(triangle)
        result.discard(vertex)
        return result

    def opposite(self, row: int, first: int, second: int, /) -> int:
        triangle = self.triangles[row]
        assert triangle is not None
        return next(v for v in triangle if v != first and v != second)

    def directed(self, row: int, first: int, /) -> _Triangle:
        """Triangle ``row`` rotated so that it starts at ``first``."""
        triangle = self.triangles[row]
        assert triangle is not None
        index = triangle.index(first)
        return (triangle[index], triangle[(index + 1) % 3], triangle[(index + 2) % 3])

    def translate(self, vertex: int, axis: int, /) -> int | None:
        """The exact upper-face copy (``+e_axis``) of a lower-plane vertex."""
        point = np.array(self.points[vertex], copy=True)
        point[axis] = self.high[axis]
        return self.at.get(point.tobytes())

    def seam_plane(self, first: int, second: int, /) -> _Plane | None:
        """The cell-face plane of a boundary (seam) edge."""
        if len(self.faces(first, second)) != 1:
            return None
        common = self.common(first, second)
        return next(iter(common)) if len(common) == 1 else None

    # ------------------------------------------------------------- geometry

    def length_of(self, first: int, second: int, /) -> float:
        return float(np.linalg.norm(self.points[second] - self.points[first]))

    def score(
        self, triangles: list[_Triangle], moved: dict[int, np.ndarray] | None = None, /
    ) -> tuple[float, float]:
        """Minimum angle and minimum normal/gradient cosine of triangles."""
        angle, alignment = 180.0, 1.0
        for triangle in triangles:
            corners = np.stack(
                [
                    self.points[v] if moved is None or v not in moved else moved[v]
                    for v in triangle
                ]
            )
            normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
            norm = float(np.linalg.norm(normal))
            if norm == 0.0:
                return 0.0, -1.0
            angle = min(angle, _minimum_angle(corners))
            for vertex in triangle:
                gradient = self.gradients[vertex]
                cosine = float(np.dot(normal, gradient)) / (
                    norm * float(np.linalg.norm(gradient))
                )
                alignment = min(alignment, cosine)
        return angle, alignment

    def admissible(
        self,
        old: list[_Triangle],
        new: list[_Triangle],
        moved: dict[int, np.ndarray] | None = None,
        /,
    ) -> tuple[bool, float, float]:
        """Orientation test of replaced triangles; returns old and new angles."""
        old_angle, old_alignment = self.score(old)
        new_angle, new_alignment = self.score(new, moved)
        aligned = new_alignment > 0.0 and new_alignment >= min(_ALIGNMENT, old_alignment)
        return aligned, old_angle, new_angle

    def inside(self, point: np.ndarray, kept: int, /) -> bool:
        """Strictly inside the open box, except the exactly kept axis."""
        return all(
            axis == kept or self.low[axis] < point[axis] < self.high[axis]
            for axis in range(3)
        )

    def plane_edges_ok(self, triangles: list[_Triangle], /) -> bool:
        """Edges joining two vertices of one plane must stay boundary (seam) edges."""
        counts: dict[tuple[int, int], int] = {}
        for triangle in triangles:
            for index in range(3):
                first, second = triangle[index], triangle[(index + 1) % 3]
                key = (min(first, second), max(first, second))
                counts[key] = counts.get(key, 0) + 1
        for (first, second), count in counts.items():
            common = self.common(first, second)
            if len(common) > 1 or (common and count != 1):
                return False
        return True

    # ------------------------------------------------------------- splits

    def split_pass(self) -> bool:
        candidates: list[tuple[float, tuple[int, int], int]] = []
        for (first, second), faces in self.edges.items():
            length = self.length_of(first, second)
            if length <= _LONG * self.length:
                continue
            plane = self.seam_plane(first, second) if len(faces) == 1 else None
            if len(faces) == 1 and (plane is None or plane[1] or self.seams_frozen):
                continue
            candidates.append((-length, (first, second), -1 if plane is None else plane[0]))
        if not candidates:
            return False
        candidates.sort()
        starts = np.stack([0.5 * (self.points[a] + self.points[b]) for _, (a, b), _ in candidates])
        axes = np.asarray([axis for _, _, axis in candidates], dtype=np.int64)
        projected, errors, gradients, certified = self.zero_set.project(
            starts, axes, 0.25 * self.length
        )
        changed = False
        for row, (_, (first, second), axis) in enumerate(candidates):
            if not certified[row] or not self.faces(first, second):
                continue
            point = (projected[row], gradients[row], float(errors[row]))
            if axis < 0:
                changed |= self.split_interior(first, second, point)
            else:
                changed |= self.split_seam(first, second, axis, point)
        return changed

    def _split_plan(
        self, first: int, second: int, new: int, /
    ) -> tuple[list[int], list[_Triangle], list[_Triangle]]:
        rows = list(self.faces(first, second))
        old: list[_Triangle] = []
        added: list[_Triangle] = []
        for row in rows:
            triangle = self.triangles[row]
            assert triangle is not None
            a, b, c = self.directed(row, self.opposite(row, first, second))
            old.append(triangle)
            added.extend(((a, b, new), (a, new, c)))
        return rows, old, added

    def split_interior(
        self, first: int, second: int, point: tuple[np.ndarray, np.ndarray, float], /
    ) -> bool:
        if not self.inside(point[0], -1):
            return False
        new = self.vertex(*point)
        rows, old, added = self._split_plan(first, second, new)
        aligned, _, _ = self.admissible(old, added)
        if not aligned or not self.plane_edges_ok(added):
            self._discard(new)
            return False
        self.replace(rows, added)
        return True

    def split_seam(
        self,
        first: int,
        second: int,
        axis: int,
        point: tuple[np.ndarray, np.ndarray, float],
        /,
    ) -> bool:
        """Split a lower seam edge and its upper translate at one certified point."""
        upper_first, upper_second = self.translate(first, axis), self.translate(second, axis)
        if (
            upper_first is None
            or upper_second is None
            or len(self.faces(upper_first, upper_second)) != 1
            or not self.inside(point[0], axis)
        ):
            return False
        moved = np.array(point[0], copy=True)
        moved[axis] = self.high[axis]
        lower = self.vertex(*point)
        upper = self.vertex(moved, point[1], point[2])
        plans = (
            self._split_plan(first, second, lower),
            self._split_plan(upper_first, upper_second, upper),
        )
        if not all(
            self.admissible(old, added)[0] and self.plane_edges_ok(added)
            for _, old, added in plans
        ):
            self._discard(upper)
            self._discard(lower)
            return False
        for rows, _, added in plans:
            self.replace(rows, added)
        return True

    def _discard(self, vertex: int, /) -> None:
        """Forget an unused, most recently added vertex."""
        assert vertex == len(self.points) - 1 and not self.stars[vertex]
        if self.planes[vertex]:
            del self.at[self.points[vertex].tobytes()]
        for bank in (self.points, self.gradients, self.errors, self.planes, self.alive):
            bank.pop()
        self.stars.pop()

    # ------------------------------------------------------------- collapses

    def collapse_pass(self) -> bool:
        candidates = sorted(
            (self.length_of(first, second), (first, second))
            for first, second in self.edges
            if self.length_of(first, second) < _SHORT * self.length
        )
        changed = False
        for _, (first, second) in candidates:
            if not self.faces(first, second):
                continue
            if self.length_of(first, second) >= _SHORT * self.length:
                continue
            changed |= self.collapse(first, second) or self.collapse(second, first)
        return changed

    def _collapse_plan(
        self, removed: int, kept: int, /
    ) -> tuple[list[int], list[_Triangle], list[_Triangle]] | None:
        shared = self.faces(removed, kept)
        link = {self.opposite(row, removed, kept) for row in shared}
        if self.neighbors(removed) & self.neighbors(kept) != link:
            return None
        rows = sorted(self.stars[removed])
        old: list[_Triangle] = []
        added: list[_Triangle] = []
        for row in rows:
            triangle = self.triangles[row]
            assert triangle is not None
            old.append(triangle)
            if kept not in triangle:
                added.append(tuple(kept if v == removed else v for v in triangle))  # type: ignore[arg-type]
        for vertex in self.neighbors(removed) - {kept}:
            if self.length_of(kept, vertex) > _LONG * self.length:
                return None
        context = sorted(self.stars[kept] - set(rows))
        neighborhood = [self.triangles[row] for row in context]
        if not self.plane_edges_ok(added + [t for t in neighborhood if t is not None]):
            return None
        return rows, old, added

    def _collapse_ok(self, plan: tuple[list[int], list[_Triangle], list[_Triangle]]) -> bool:
        _, old, added = plan
        aligned, old_angle, new_angle = self.admissible(old, added)
        return aligned and new_angle >= min(old_angle, _COLLAPSE_ANGLE)

    def collapse(self, removed: int, kept: int, /) -> bool:
        planes = self.planes[removed]
        if len(planes) > 1:
            return False
        plans = []
        if planes:
            axis, side = next(iter(planes))
            if (
                side
                or self.seams_frozen
                or self.seam_plane(removed, kept) != (axis, side)
            ):
                return False
            upper_removed = self.translate(removed, axis)
            upper_kept = self.translate(kept, axis)
            if upper_removed is None or upper_kept is None:
                return False
            pairs = ((removed, kept), (upper_removed, upper_kept))
        else:
            pairs = ((removed, kept),)
        for pair in pairs:
            plan = self._collapse_plan(*pair)
            if plan is None or not self._collapse_ok(plan):
                return False
            plans.append(plan)
        for (gone, _), (rows, _, added) in zip(pairs, plans, strict=True):
            self.replace(rows, added)
            self.alive[gone] = False
            if self.planes[gone]:
                del self.at[self.points[gone].tobytes()]
        return True

    # ------------------------------------------------------------- flips

    def flip_pass(self) -> bool:
        changed = False
        for first, second in sorted(self.edges):
            faces = self.faces(first, second)
            if len(faces) != 2:
                continue
            changed |= self.flip(first, second)
        return changed

    def flip(self, first: int, second: int, /) -> bool:
        rows = list(self.faces(first, second))
        a, b, c = self.directed(rows[0], first)
        if b != second:
            rows.reverse()
            a, b, c = self.directed(rows[0], first)
        d = self.opposite(rows[1], first, second)
        if c == d or self.faces(c, d) or self.common(c, d):
            return False
        old = [(a, b, c), (b, a, d)]
        added = [(a, d, c), (d, b, c)]
        aligned, old_angle, new_angle = self.admissible(old, added)
        if not aligned or new_angle <= old_angle + _GAIN:
            return False
        self.replace(rows, added)
        return True

    # ------------------------------------------------------------- relaxation

    def relax_pass(self) -> bool:
        movers: list[int] = []
        targets: list[np.ndarray] = []
        axes: list[int] = []
        for vertex in range(len(self.points)):
            target = self._relax_target(vertex) if self.alive[vertex] else None
            if target is not None:
                movers.append(vertex)
                targets.append(target[0])
                axes.append(target[1])
        if not movers:
            return False
        projected, errors, gradients, certified = self.zero_set.project(
            np.stack(targets), np.asarray(axes, dtype=np.int64), 0.25 * self.length
        )
        changed = False
        for row, vertex in enumerate(movers):
            if certified[row]:
                changed |= self.move(
                    vertex, axes[row], (projected[row], gradients[row], float(errors[row]))
                )
        return changed

    def _relax_target(self, vertex: int, /) -> tuple[np.ndarray, int] | None:
        planes = self.planes[vertex]
        if not planes:
            weights, centre = 0.0, np.zeros(3)
            for row in self.stars[vertex]:
                triangle = self.triangles[row]
                assert triangle is not None
                corners = np.stack([self.points[v] for v in triangle])
                area = 0.5 * float(
                    np.linalg.norm(np.cross(corners[1] - corners[0], corners[2] - corners[0]))
                )
                weights += area
                centre += area * corners.mean(axis=0)
            if weights == 0.0:
                return None
            normal = self.gradients[vertex] / np.linalg.norm(self.gradients[vertex])
            shift = centre / weights - self.points[vertex]
            return self.points[vertex] + shift - np.dot(shift, normal) * normal, -1
        if len(planes) > 1 or self.seams_frozen:
            return None
        axis, side = next(iter(planes))
        ends = [
            other
            for other in self.neighbors(vertex)
            if self.seam_plane(vertex, other) == (axis, side)
        ]
        if side or len(ends) != 2:
            return None
        target = 0.5 * (self.points[ends[0]] + self.points[ends[1]])
        target[axis] = self.low[axis]
        return target, axis

    def move(
        self, vertex: int, axis: int, point: tuple[np.ndarray, np.ndarray, float], /
    ) -> bool:
        if not self.inside(point[0], axis):
            return False
        copies = [(vertex, point)]
        if axis >= 0:
            upper = self.translate(vertex, axis)
            if upper is None:
                return False
            moved = np.array(point[0], copy=True)
            moved[axis] = self.high[axis]
            copies.append((upper, (moved, point[1], point[2])))
        for copy, (position, _, _) in copies:
            star = [self.triangles[row] for row in sorted(self.stars[copy])]
            triangles = [t for t in star if t is not None]
            old_angle, old_alignment = self.score(triangles)
            new_angle, new_alignment = self.score(triangles, {copy: position})
            if (
                new_alignment <= 0.0
                or new_alignment < min(_ALIGNMENT, old_alignment)
                or new_angle <= old_angle + _GAIN
            ):
                return False
        for copy, (position, gradient, error) in copies:
            if self.planes[copy]:
                del self.at[self.points[copy].tobytes()]
            self.points[copy] = np.array(position, dtype=np.float64)
            self.gradients[copy] = np.array(gradient, dtype=np.float64)
            self.errors[copy] = error
            if self.planes[copy]:
                self.at[self.points[copy].tobytes()] = copy
        return True

    def remesh(self) -> None:
        """Run split, collapse, flip and relaxation passes until none changes."""
        while True:
            changed = self.split_pass()
            changed |= self.collapse_pass()
            changed |= self.flip_pass()
            changed |= self.relax_pass()
            if not changed:
                return


@dataclass(slots=True)
class _Face:
    """Constrained cap input of one lower cell face, in its plane coordinates."""

    points: list[np.ndarray]
    references: list[tuple[str, object]]
    segments: list[tuple[int, int]]
    seams: dict[int, tuple[int, int]]
    sides: dict[int, tuple[int, int, int]]


class _Caps:
    """Cap triangulations of the lower cell faces and their box-edge samples."""

    def __init__(self, patch: _Patch, /) -> None:
        self.patch = patch
        low = patch.low
        self.corner_inside = int(patch.zero_set.signs(low[None, :])[0]) < 0
        if not patch.zero_set.signs(low[None, :])[0]:
            raise ValueError("The fundamental box corner lies on the zero set.")
        # Coordinates along each canonical box edge: field crossings and samples.
        self.samples: list[list[float]] = [self._initial_samples(axis) for axis in range(3)]

    def crossings(self, axis: int, /) -> list[float]:
        """Box-edge roots of the canonical edge along ``axis`` (both others low)."""
        others = {(other, 0) for other in range(3) if other != axis}
        return sorted(
            float(self.patch.points[v][axis])
            for v in range(len(self.patch.points))
            if self.patch.alive[v] and set(self.patch.planes[v]) == others
        )

    def _inside_intervals(self, axis: int, /) -> list[tuple[float, float]]:
        stops = [float(self.patch.low[axis]), *self.crossings(axis), float(self.patch.high[axis])]
        inside = self.corner_inside
        intervals = []
        for start, stop in zip(stops[:-1], stops[1:], strict=True):
            if inside:
                intervals.append((start, stop))
            inside = not inside
        return intervals

    def _initial_samples(self, axis: int, /) -> list[float]:
        samples: list[float] = []
        for start, stop in self._inside_intervals(axis):
            pieces = max(1, math.ceil((stop - start) / self.patch.length))
            samples.extend(start + (stop - start) * k / pieces for k in range(1, pieces))
        return samples

    def edge_point(self, axis: int, value: float, fixed: dict[int, int], /) -> np.ndarray:
        point = np.array(self.patch.low, copy=True)
        point[axis] = value
        for other, side in fixed.items():
            point[other] = self.patch.high[other] if side else self.patch.low[other]
        return point

    def face(self, axis: int, /) -> _Face:
        """Points and segments of the region's trace on the lower face ``axis``."""
        patch = self.patch
        first, second = (axis + 1) % 3, (axis + 2) % 3
        face = _Face([], [], [], {}, {})
        index: dict[bytes, int] = {}

        def add(point: np.ndarray, reference: tuple[str, object], /) -> int:
            key = point.tobytes()
            if key not in index:
                index[key] = len(face.points)
                face.points.append(np.asarray((point[first], point[second])))
                face.references.append(reference)
            return index[key]

        for vertex in range(len(patch.points)):
            if patch.alive[vertex] and (axis, 0) in patch.planes[vertex]:
                add(patch.points[vertex], ("patch", vertex))
        for (a, b), faces in patch.edges.items():
            if len(faces) == 1 and patch.seam_plane(a, b) == (axis, 0):
                row = len(face.segments)
                face.segments.append((index[patch.points[a].tobytes()], index[patch.points[b].tobytes()]))
                face.seams[row] = (a, b)
        for along, across in ((first, second), (second, first)):
            for side in (0, 1):
                self._side(face, add, axis, along, across, side)
        return face

    def _side(self, face: _Face, add, axis: int, along: int, across: int, side: int) -> None:
        """Inside pieces of one face-square side (a copy of the canonical box edge)."""
        fixed = {axis: 0, across: side}
        stops = [float(self.patch.low[along]), *self.crossings(along), float(self.patch.high[along])]
        crossings = set(stops[1:-1])
        values = sorted({*stops, *self.samples[along]})
        inside = self.corner_inside
        previous: int | None = None
        for value in values:
            point = self.edge_point(along, value, fixed)
            reference: tuple[str, object] = (
                ("patch", self.patch.at[point.tobytes()])
                if value in crossings
                else ("point", point)
            )
            current = add(point, reference) if inside or value in crossings else None
            if inside and previous is not None and current is not None:
                face.sides[len(face.segments)] = (along, across, side)
                face.segments.append((previous, current))
            if value in crossings:
                inside = not inside
            previous = current if inside else None

    def triangulate(self, axis: int, /) -> tuple[_Face, np.ndarray] | list[tuple]:
        """Refined cap triangulation, or the constraint splits it requests."""
        face = self.face(axis)
        points = np.asarray(face.points, dtype=np.float64)
        segments = np.asarray(face.segments, dtype=np.int64).reshape(-1, 2)
        holes = self._holes(axis, points, segments)
        result = constrained_delaunay_2d(
            points,
            segments,
            holes=holes,
            min_angle=_CAP_MINIMUM_ANGLE,
            max_area=math.sqrt(3.0) / 4.0 * self.patch.length**2,
        )
        steiner = np.flatnonzero(
            np.asarray(result.constraint_sources)[points.shape[0] :] >= 0
        ) + points.shape[0]
        requests = [
            (int(result.constraint_sources[row]), np.asarray(result.points[row]))
            for row in steiner.tolist()
        ]
        if requests:
            return [(face, segment, point) for segment, point in requests]
        return face, self._triangles(face, result)

    def _holes(self, axis: int, points: np.ndarray, segments: np.ndarray) -> np.ndarray:
        """One seed inside each constrained component of the face outside the region."""
        result = constrained_delaunay_2d(points, segments, keep_convex_hull=True)
        triangles = np.asarray(result.triangles, dtype=np.int64)
        constrained = {(min(a, b), max(a, b)) for a, b in segments.tolist()}
        parent = list(range(triangles.shape[0]))

        def find(item: int) -> int:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        owner: dict[tuple[int, int], int] = {}
        for row, triangle in enumerate(triangles.tolist()):
            for index in range(3):
                a, b = triangle[index], triangle[(index + 1) % 3]
                key = (min(a, b), max(a, b))
                if key in constrained:
                    continue
                other = owner.setdefault(key, row)
                if other != row:
                    parent[find(other)] = find(row)
        corners = np.asarray(result.points, dtype=np.float64)[triangles]
        centroids = corners.mean(axis=1)
        sides = np.stack((corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
        areas = np.abs(sides[:, 0, 0] * sides[:, 1, 1] - sides[:, 0, 1] * sides[:, 1, 0])
        perimeters = sum(
            np.linalg.norm(corners[:, (k + 1) % 3] - corners[:, k], axis=1) for k in range(3)
        )
        radii = areas / np.maximum(perimeters, np.finfo(np.float64).tiny)
        lifted = np.tile(self.patch.low, (triangles.shape[0], 1))
        lifted[:, (axis + 1) % 3] = centroids[:, 0]
        lifted[:, (axis + 2) % 3] = centroids[:, 1]
        signs = self.patch.zero_set.signs(lifted)
        components: dict[int, list[int]] = {}
        for row in range(triangles.shape[0]):
            components.setdefault(find(row), []).append(row)
        holes = []
        for rows in components.values():
            ordered = sorted(rows, key=lambda row: -radii[row])
            decided = next((row for row in ordered if signs[row] != 0), None)
            if decided is None:
                raise ValueError("A cap component has no certified side.")
            if signs[decided] > 0:
                holes.append(centroids[decided])
        return np.asarray(holes, dtype=np.float64).reshape(-1, 2)

    @staticmethod
    def _triangles(face: _Face, result: object) -> np.ndarray:
        points = np.asarray(result.points, dtype=np.float64)  # type: ignore[attr-defined]
        triangles = np.asarray(result.triangles, dtype=np.int64)  # type: ignore[attr-defined]
        corners = points[triangles]
        orientation = (corners[:, 1, 0] - corners[:, 0, 0]) * (
            corners[:, 2, 1] - corners[:, 0, 1]
        ) - (corners[:, 1, 1] - corners[:, 0, 1]) * (corners[:, 2, 0] - corners[:, 0, 0])
        if np.any(orientation == 0.0):
            raise ValueError("A cap triangle is degenerate.")
        triangles[orientation < 0.0] = triangles[orientation < 0.0][:, ::-1]
        for row in range(len(face.points), points.shape[0]):
            face.points.append(points[row])
            face.references.append(("interior", row))
        return triangles

    def request(self, face: _Face, axis: int, segment: int, point: np.ndarray) -> None:
        """Turn a refinement point on a constraint into a seam split or a sample."""
        patch = self.patch
        if segment in face.seams:
            a, b = face.seams[segment]
            start = np.array(patch.low, copy=True)
            start[(axis + 1) % 3], start[(axis + 2) % 3] = point
            projected, errors, gradients, certified = patch.zero_set.project(
                start[None, :], np.asarray([axis]), 0.5 * patch.length_of(a, b)
            )
            if not certified[0] or not patch.split_seam(
                a, b, axis, (projected[0], gradients[0], float(errors[0]))
            ):
                raise ValueError("A cap refinement point on a seam has no certified split.")
            return
        along, _, _ = face.sides[segment]
        value = float(point[0] if along == (axis + 1) % 3 else point[1])
        if value not in self.samples[along]:
            self.samples[along].append(value)
            self.samples[along].sort()


def _surface_edge_length(points: np.ndarray, triangles: np.ndarray, /) -> float:
    edges = np.unique(
        np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1),
        axis=0,
    )
    return float(np.mean(np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)))


def periodic_box_boundary(
    surface: PeriodicImplicitSurface, edge_length: float | None, /
) -> PeriodicBoxBoundary:
    """Remeshed contour and refined cell-face caps of a resolved discovery."""
    points = np.asarray(surface.reduced_points, dtype=np.float64)
    triangles = np.asarray(surface.faces, dtype=np.int64)
    length = _surface_edge_length(points, triangles) if edge_length is None else edge_length
    zero_set = QuotientZeroSet(surface)
    patch = _Patch(
        zero_set, np.asarray(surface.reduced_domain), points, triangles, surface.root_error, length
    )
    patch.remesh()
    caps = _Caps(patch)
    while True:
        outcomes = [caps.triangulate(axis) for axis in range(3)]
        requests = [
            (axis, item) for axis, outcome in enumerate(outcomes) if isinstance(outcome, list)
            for item in outcome
        ]
        if not requests:
            break
        patch.seams_frozen = True
        for axis, (face, segment, point) in requests:
            caps.request(face, axis, segment, point)
        patch.remesh()
    return _assemble(patch, caps, outcomes)  # type: ignore[arg-type]


def _assemble(
    patch: _Patch, caps: _Caps, outcomes: list[tuple[_Face, np.ndarray]], /
) -> PeriodicBoxBoundary:
    low, high = patch.low, patch.high
    rows: dict[bytes, int] = {}
    points: list[np.ndarray] = []

    def row_of(point: np.ndarray, /) -> int:
        key = np.asarray(point, dtype=np.float64).tobytes()
        if key not in rows:
            rows[key] = len(points)
            points.append(np.asarray(point, dtype=np.float64))
        return rows[key]

    surface = [
        tuple(row_of(patch.points[v]) for v in triangle)
        for triangle in patch.triangles
        if triangle is not None
    ]
    contour_rows = sorted(set(row for triangle in surface for row in triangle))
    error = max(
        patch.errors[v] for v in range(len(patch.points)) if patch.alive[v] and patch.stars[v]
    )
    cap_pairs = []
    for axis, (face, triangles) in enumerate(outcomes):
        lifted = []
        for reference, plane in zip(face.references, face.points, strict=True):
            kind, value = reference
            if kind == "patch":
                lifted.append(np.asarray(patch.points[int(value)]))  # type: ignore[arg-type]
            elif kind == "point":
                lifted.append(np.asarray(value))
            else:
                point = np.array(low, copy=True)
                point[(axis + 1) % 3], point[(axis + 2) % 3] = plane
                lifted.append(point)
        lower = np.asarray([[row_of(lifted[v]) for v in t] for t in triangles.tolist()])
        moved = []
        for point in lifted:
            translate = np.array(point, copy=True)
            translate[axis] = high[axis]
            moved.append(translate)
        upper = np.asarray([[row_of(moved[v]) for v in t] for t in triangles.tolist()])
        cap_pairs.append((lower.reshape(-1, 3), upper.reshape(-1, 3)))
    lifted_points = np.asarray(points, dtype=np.float64)
    windings = (lifted_points == high[None, :]).astype(np.int64)
    canonical = np.where(windings == 1, low[None, :], lifted_points)
    orbit_ids: dict[bytes, int] = {}
    orbits = np.asarray(
        [orbit_ids.setdefault(row.tobytes(), len(orbit_ids)) for row in canonical],
        dtype=np.int64,
    )
    del contour_rows
    return PeriodicBoxBoundary(
        lifted_points,
        orbits,
        windings,
        np.asarray(surface, dtype=np.int64).reshape(-1, 3),
        tuple(cap_pairs),
        float(error),
    )
