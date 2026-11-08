#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Well-shaped boundary of a periodic trigonometric region in a fundamental box.

The discovery's lifted contour (marching tetrahedra on the target-level grid)
has needles and caps wherever a root sits near a grid corner. Its cell-face
caps, tiled by grid squares cut at those roots, inherit them. A fixed-boundary
fill keeps every boundary triangle, so those triangles force bad tetrahedra.
The box itself is an artifact of the fundamental domain in the three-torus:
where its faces cut the surface at a shallow crease, or its edges cross the
surface at a shallow angle, the fill's tetrahedra sit in thin wedges whose
circumcentres leave the region, and the fixed boundary blocks every repair.
This module rebuilds the boundary before the fill:

* the fundamental box is chosen by the region's sampled wedge angles along
  the cell-face traces and at the box-edge crossings, and the discovery's
  quotient contour is lifted into it (`quotient_box_contour`).
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
is replaced by the lower one, which is exact because every box offset keeps
``offset + 1`` exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .._meshcore import (
    constrained_delaunay_2d,
    ConstrainedDelaunay2D,
    MeshcoreStatus,
)
from ..geometry.implicit._periodic_discovery import (
    PeriodicImplicitSurface,
    quotient_box_contour,
    QuotientBoxContour,
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
_GAIN = 0.1
# Growth of the target edge length per unit distance from a short frozen seam edge.
_GRADATION = 0.5
# Candidate box offsets per axis: odd multiples of 1/64. They keep offset + 1
# exact and miss the quarter-lattice points where symmetric tables put their
# zero set on the discovery grid. The published offset is a candidate too.
_OFFSETS = (2.0 * np.arange(32, dtype=np.float64) + 1.0) / 64.0
# Samples per side of a candidate face.
_FACE_SAMPLES = 128


@dataclass(frozen=True, slots=True)
class PeriodicBoxBoundary:
    """Lifted boundary of the region in the fundamental box ``domain``.

    ``points`` are lifted reduced coordinates with ``orbits`` (canonical
    reduced-coordinate classes) and ``windings``. ``surface`` holds the outward
    contour triangles. ``caps[axis]`` is the pair of lower and upper cap
    triangulations of that axis, both counter-clockwise about ``+e_axis``; the
    upper one is the exact translate of the lower one. ``root_error`` bounds
    the reduced distance of every contour vertex from a root of the field, and
    ``interior`` is a reduced point of the box certified inside the region.
    """

    domain: np.ndarray
    points: np.ndarray
    orbits: np.ndarray
    windings: np.ndarray
    surface: np.ndarray
    caps: tuple[tuple[np.ndarray, np.ndarray], ...]
    root_error: float
    interior: np.ndarray


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
        self.centers = np.empty((0, 3), dtype=np.float64)
        self.targets = np.empty(0, dtype=np.float64)
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

    def triangle(self, row: int, /) -> _Triangle:
        """The live triangle ``row``; a removed row is an internal-state error."""
        triangle = self.triangles[row]
        if triangle is None:
            raise RuntimeError("A removed contour triangle is still referenced.")
        return triangle

    def _remove(self, row: int, /) -> None:
        triangle = self.triangle(row)
        self.triangles[row] = None
        for index in range(3):
            first, second = triangle[index], triangle[(index + 1) % 3]
            key = (min(first, second), max(first, second))
            faces = self.edges[key]
            faces.remove(row)
            if not faces:
                del self.edges[key]
            self.stars[first].discard(row)

    def vertex(self, point: np.ndarray, gradient: np.ndarray, error: float, /) -> int:
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
            result.update(self.triangle(row))
        result.discard(vertex)
        return result

    def opposite(self, row: int, first: int, second: int, /) -> int:
        return next(v for v in self.triangle(row) if v != first and v != second)

    def directed(self, row: int, first: int, /) -> _Triangle:
        """Triangle ``row`` rotated so that it starts at ``first``."""
        triangle = self.triangle(row)
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
        """Minimum angle (degrees) and minimum normal/gradient cosine of triangles."""
        if not triangles:
            return 180.0, 1.0
        rows = np.asarray(triangles, dtype=np.int64)
        corners = np.stack(
            [
                self.points[v] if moved is None or v not in moved else moved[v]
                for v in rows.ravel().tolist()
            ]
        ).reshape(-1, 3, 3)
        gradients = np.stack([self.gradients[v] for v in rows.ravel().tolist()]).reshape(
            -1, 3, 3
        )
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        norms = np.linalg.norm(normals, axis=1)
        sides = np.roll(corners, -1, axis=1) - corners
        lengths = np.linalg.norm(sides, axis=2)
        if np.any(norms == 0.0) or np.any(lengths == 0.0):
            return 0.0, -1.0
        # The angle at corner k lies between the sides k and k - 1.
        cosines = -np.sum(sides * np.roll(sides, 1, axis=1), axis=2) / (
            lengths * np.roll(lengths, 1, axis=1)
        )
        angle = float(np.degrees(np.arccos(np.clip(np.max(cosines), -1.0, 1.0))))
        alignment = np.sum(normals[:, None, :] * gradients, axis=2) / (
            norms[:, None] * np.linalg.norm(gradients, axis=2)
        )
        return angle, float(np.min(alignment))

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

    def sizes(self, points: np.ndarray, /) -> np.ndarray:
        """Target edge lengths at points, graded away from short frozen seam edges.

        A seam edge of length ``s`` and midpoint ``p`` asks for edges no longer
        than ``s + _GRADATION * d`` at distance ``d`` from ``p``, so the surface
        coarsens smoothly from refined seam pieces toward ``length``.
        """
        values = np.full(points.shape[0], self.length)
        if not self.targets.size or not points.shape[0]:
            return values
        centers, targets = self.centers, self.targets
        distance = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        graded = np.min(targets[None, :] + _GRADATION * distance, axis=1)
        return np.minimum(values, graded)

    def _edge_sizes(
        self, edges: list[tuple[int, int]], /
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lengths of edges and the target length at their midpoints."""
        if not edges:
            return np.empty(0), np.empty(0)
        first = np.stack([self.points[a] for a, _ in edges])
        second = np.stack([self.points[b] for _, b in edges])
        lengths = np.linalg.norm(second - first, axis=1)
        return lengths, self.sizes(0.5 * (first + second))

    def freeze_seams(self) -> None:
        """Keep the seam polyline fixed and grade the surface from its short edges."""
        self.seams_frozen = True
        seams = [edge for edge, faces in self.edges.items() if len(faces) == 1]
        lengths, _ = self._edge_sizes(seams)
        short = [
            (edge, float(length))
            for edge, length in zip(seams, lengths, strict=True)
            if length < self.length
        ]
        # Built once per freeze: ``sizes`` runs for every candidate edge.
        self.centers = np.asarray(
            [0.5 * (self.points[a] + self.points[b]) for (a, b), _ in short],
            dtype=np.float64,
        ).reshape(-1, 3)
        self.targets = np.asarray([length for _, length in short], dtype=np.float64)

    def split_pass(self) -> bool:
        candidates: list[tuple[float, tuple[int, int], int]] = []
        edges = list(self.edges)
        lengths, sizes = self._edge_sizes(edges)
        for (first, second), length, size in zip(edges, lengths, sizes, strict=True):
            if length <= _LONG * size:
                continue
            faces = self.edges[(first, second)]
            plane = self.seam_plane(first, second) if len(faces) == 1 else None
            if len(faces) == 1 and (plane is None or plane[1] or self.seams_frozen):
                continue
            candidates.append(
                (-float(length), (first, second), -1 if plane is None else plane[0])
            )
        if not candidates:
            return False
        candidates.sort()
        starts = np.stack(
            [0.5 * (self.points[a] + self.points[b]) for _, (a, b), _ in candidates]
        )
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
            triangle = self.triangle(row)
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
        upper_first, upper_second = (
            self.translate(first, axis),
            self.translate(second, axis),
        )
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
        if vertex != len(self.points) - 1 or self.stars[vertex]:
            raise RuntimeError("Only an unused, most recently added vertex is discarded.")
        if self.planes[vertex]:
            del self.at[self.points[vertex].tobytes()]
        for bank in (self.points, self.gradients, self.errors, self.planes, self.alive):
            bank.pop()
        self.stars.pop()

    # ------------------------------------------------------------- collapses

    def collapse_pass(self) -> bool:
        edges = list(self.edges)
        lengths, sizes = self._edge_sizes(edges)
        candidates = sorted(
            (float(length), edge)
            for edge, length, size in zip(edges, lengths, sizes, strict=True)
            if length < _SHORT * size
        )
        changed = False
        for _, (first, second) in candidates:
            if not self.faces(first, second):
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
            triangle = self.triangle(row)
            old.append(triangle)
            if kept not in triangle:
                a, b, c = (kept if v == removed else v for v in triangle)
                added.append((a, b, c))
        others = sorted(self.neighbors(removed) - {kept})
        lengths, sizes = self._edge_sizes([(kept, vertex) for vertex in others])
        if np.any(lengths > _LONG * sizes):
            return None
        context = sorted(self.stars[kept] - set(rows))
        neighborhood = [self.triangle(row) for row in context]
        if not self.plane_edges_ok(added + neighborhood):
            return None
        return rows, old, added

    def _collapse_ok(
        self, plan: tuple[list[int], list[_Triangle], list[_Triangle]]
    ) -> bool:
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
        # A flip never creates an edge the split pass would split again.
        length, size = self._edge_sizes([(c, d)])
        if length[0] > _LONG * size[0]:
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
            # Only a target that already raises the star's smallest angle is
            # worth its certified projection.
            if target is not None and self._improves(vertex, target[0]):
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
                    vertex,
                    axes[row],
                    (projected[row], gradients[row], float(errors[row])),
                )
        return changed

    def _improves(self, vertex: int, position: np.ndarray, /) -> bool:
        """Whether moving ``vertex`` to ``position`` raises its star's smallest angle."""
        triangles = [self.triangle(row) for row in sorted(self.stars[vertex])]
        return (
            self.score(triangles, {vertex: position})[0]
            > self.score(triangles)[0] + _GAIN
        )

    def _relax_target(self, vertex: int, /) -> tuple[np.ndarray, int] | None:
        planes = self.planes[vertex]
        if not planes:
            weights, centre = 0.0, np.zeros(3)
            for row in self.stars[vertex]:
                corners = np.stack([self.points[v] for v in self.triangle(row)])
                area = 0.5 * float(
                    np.linalg.norm(
                        np.cross(corners[1] - corners[0], corners[2] - corners[0])
                    )
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
            triangles = [self.triangle(row) for row in sorted(self.stars[copy])]
            old_angle, old_alignment = self.score(triangles)
            new_angle, new_alignment = self.score(triangles, {copy: position})
            if (
                new_alignment <= 0.0
                or new_alignment < min(_ALIGNMENT, old_alignment)
                or new_angle <= old_angle + _GAIN
                or self._lengthens(copy, position)
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

    def _lengthens(self, vertex: int, position: np.ndarray, /) -> bool:
        """Whether moving ``vertex`` makes an incident edge newly or more too long.

        Relaxation never creates work for the split pass, so splits stop once
        no edge is too long and the passes reach their fixpoint.
        """
        others = np.asarray([self.points[v] for v in sorted(self.neighbors(vertex))])
        old = np.linalg.norm(others - self.points[vertex], axis=1)
        new = np.linalg.norm(others - position, axis=1)
        sizes = self.sizes(0.5 * (others + position))
        return bool(np.any(new > np.maximum(old, _LONG * sizes)))

    def remesh(self) -> None:
        """Split, then collapse, then flip and relax, each phase to its fixpoint.

        Every split shortens a too-long edge, every collapse removes a vertex
        without creating a too-long edge, and every flip or move strictly
        raises the smallest angle of the triangles it replaces without
        lengthening an edge past its target. No phase undoes an earlier one.
        """
        while self.split_pass():
            pass
        while self.collapse_pass():
            pass
        while self.flip_pass() | self.relax_pass():
            pass


@dataclass(slots=True)
class _Face:
    """Constrained cap input of one lower cell face.

    ``points`` are plane coordinates ``(u[axis + 1], u[axis + 2])`` and
    ``lifted`` the exact reduced points they stand for. ``seams`` maps a
    segment to its contour seam edge and ``sides`` maps a face-square segment
    to its box edge ``(along, across, side)``.
    """

    axis: int
    points: list[np.ndarray]
    lifted: list[np.ndarray]
    segments: list[tuple[int, int]]
    seams: dict[int, tuple[int, int]]
    sides: dict[int, tuple[int, int, int]]

    def add(self, point: np.ndarray, index: dict[bytes, int], /) -> int:
        """Row of a reduced point on the face, added once."""
        key = point.tobytes()
        if key not in index:
            index[key] = len(self.points)
            self.points.append(
                np.asarray((point[(self.axis + 1) % 3], point[(self.axis + 2) % 3]))
            )
            self.lifted.append(np.array(point, dtype=np.float64))
        return index[key]


@dataclass(frozen=True, slots=True)
class _Request:
    """A cap refinement point on a constraint of one face."""

    face: _Face
    segment: int
    point: np.ndarray


class _Caps:
    """Cap triangulations of the lower cell faces and their box-edge samples."""

    def __init__(self, patch: _Patch, /) -> None:
        self.patch = patch
        sign = int(patch.zero_set.signs(patch.low[None, :])[0])
        if not sign:
            raise ValueError("The fundamental box corner lies on the zero set.")
        self.corner_inside = sign < 0
        # Coordinates along each canonical box edge: field crossings and samples.
        self.samples: list[list[float]] = [
            self._initial_samples(axis) for axis in range(3)
        ]

    def crossings(self, axis: int, /) -> list[float]:
        """Box-edge roots of the canonical edge along ``axis`` (both others low)."""
        others = {(other, 0) for other in range(3) if other != axis}
        return sorted(
            float(self.patch.points[v][axis])
            for v in range(len(self.patch.points))
            if self.patch.alive[v] and set(self.patch.planes[v]) == others
        )

    def _initial_samples(self, axis: int, /) -> list[float]:
        """Uniform samples (spacing at most ``length``) of the edge's inside pieces."""
        stops = [
            float(self.patch.low[axis]),
            *self.crossings(axis),
            float(self.patch.high[axis]),
        ]
        samples: list[float] = []
        inside = self.corner_inside
        for start, stop in zip(stops[:-1], stops[1:], strict=True):
            if inside:
                pieces = max(1, math.ceil((stop - start) / self.patch.length))
                samples.extend(
                    start + (stop - start) * k / pieces for k in range(1, pieces)
                )
            inside = not inside
        return samples

    def face(self, axis: int, /) -> _Face:
        """Points and segments of the region's trace on the lower face ``axis``."""
        patch = self.patch
        first, second = (axis + 1) % 3, (axis + 2) % 3
        face = _Face(axis, [], [], [], {}, {})
        index: dict[bytes, int] = {}
        for vertex in range(len(patch.points)):
            if patch.alive[vertex] and (axis, 0) in patch.planes[vertex]:
                face.add(patch.points[vertex], index)
        for (a, b), faces in patch.edges.items():
            if len(faces) == 1 and patch.seam_plane(a, b) == (axis, 0):
                face.seams[len(face.segments)] = (a, b)
                face.segments.append(
                    (index[patch.points[a].tobytes()], index[patch.points[b].tobytes()])
                )
        for along, across in ((first, second), (second, first)):
            for side in (0, 1):
                self._side(face, index, along, across, side)
        return face

    def _side(
        self, face: _Face, index: dict[bytes, int], along: int, across: int, side: int, /
    ) -> None:
        """Inside pieces of one face-square side (a copy of the canonical box edge)."""
        patch = self.patch
        crossings = set(self.crossings(along))
        values = sorted(
            {
                float(patch.low[along]),
                float(patch.high[along]),
                *crossings,
                *self.samples[along],
            }
        )
        inside = self.corner_inside
        previous: int | None = None
        for value in values:
            point = np.array(patch.low, copy=True)
            point[along] = value
            point[across] = patch.high[across] if side else patch.low[across]
            if value in crossings and point.tobytes() not in patch.at:
                raise ValueError(
                    "A box-edge root has no exact copy on a paired box edge."
                )
            current = face.add(point, index) if inside or value in crossings else None
            if inside and previous is not None and current is not None:
                face.sides[len(face.segments)] = (along, across, side)
                face.segments.append((previous, current))
            if value in crossings:
                inside = not inside
            previous = current if inside else None

    def triangulate(self, axis: int, /) -> tuple[_Face, np.ndarray] | list[_Request]:
        """Refined cap triangulation, or the constraint refinements it requests."""
        face = self.face(axis)
        points = np.asarray(face.points, dtype=np.float64)
        segments = np.asarray(face.segments, dtype=np.int64).reshape(-1, 2)
        result = constrained_delaunay_2d(
            points,
            segments,
            holes=self._holes(axis, points, segments),
            min_angle=_CAP_MINIMUM_ANGLE,
            max_area=math.sqrt(3.0) / 4.0 * self.patch.length**2,
        )
        if result.status != MeshcoreStatus.OK:
            raise ValueError("A cell-face cap refinement is unresolved.")
        sources = np.asarray(result.constraint_sources, dtype=np.int64)
        requests = [
            _Request(face, int(sources[row]), np.asarray(result.points[row]))
            for row in range(points.shape[0], sources.size)
            if sources[row] >= 0
        ]
        if requests:
            return requests
        return face, self._triangles(face, result)

    def _holes(
        self, axis: int, points: np.ndarray, segments: np.ndarray, /
    ) -> np.ndarray:
        """One seed inside each constrained component of the face outside the region."""
        result = constrained_delaunay_2d(points, segments, keep_convex_hull=True)
        triangles = np.asarray(result.triangles, dtype=np.int64)
        constrained = {(min(a, b), max(a, b)) for a, b in segments.tolist()}
        parent = list(range(triangles.shape[0]))

        def find(item: int, /) -> int:
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
        sides = np.stack(
            (corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
        )
        areas = np.abs(sides[:, 0, 0] * sides[:, 1, 1] - sides[:, 0, 1] * sides[:, 1, 0])
        perimeters = np.sum(
            np.linalg.norm(corners - np.roll(corners, -1, axis=1), axis=2), axis=1
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
            # The certified sign of the component's roundest decided triangle.
            ordered = sorted(rows, key=lambda row: -radii[row])
            decided = next((row for row in ordered if signs[row] != 0), None)
            if decided is None:
                raise ValueError("A cap component has no certified side.")
            if signs[decided] > 0:
                holes.append(centroids[decided])
        return np.asarray(holes, dtype=np.float64).reshape(-1, 2)

    def _triangles(self, face: _Face, result: ConstrainedDelaunay2D, /) -> np.ndarray:
        """Counter-clockwise cap triangles; interior Steiner points join the face."""
        points = np.asarray(result.points, dtype=np.float64)
        triangles = np.asarray(result.triangles, dtype=np.int64)
        corners = points[triangles]
        orientation = (corners[:, 1, 0] - corners[:, 0, 0]) * (
            corners[:, 2, 1] - corners[:, 0, 1]
        ) - (corners[:, 1, 1] - corners[:, 0, 1]) * (corners[:, 2, 0] - corners[:, 0, 0])
        if np.any(orientation == 0.0):
            raise ValueError("A cap triangle is degenerate.")
        triangles[orientation < 0.0] = triangles[orientation < 0.0][:, ::-1]
        for row in range(len(face.points), points.shape[0]):
            lifted = np.array(self.patch.low, copy=True)
            lifted[(face.axis + 1) % 3], lifted[(face.axis + 2) % 3] = points[row]
            face.points.append(points[row])
            face.lifted.append(lifted)
        return triangles

    def request(self, request: _Request, /) -> None:
        """Turn a refinement point on a constraint into a seam split or a sample."""
        patch, face = self.patch, request.face
        axis = face.axis
        if request.segment in face.seams:
            a, b = face.seams[request.segment]
            # One split per seam edge and round: a second refinement point on
            # an edge already split is asked again by the next triangulation.
            if not patch.faces(a, b):
                return
            start = np.array(patch.low, copy=True)
            start[(axis + 1) % 3], start[(axis + 2) % 3] = request.point
            projected, errors, gradients, certified = patch.zero_set.project(
                start[None, :], np.asarray([axis]), 0.5 * patch.length_of(a, b)
            )
            if not certified[0] or not patch.split_seam(
                a, b, axis, (projected[0], gradients[0], float(errors[0]))
            ):
                raise ValueError(
                    "A cap refinement point on a seam has no certified split."
                )
            return
        along, _, _ = face.sides[request.segment]
        value = float(request.point[0] if along == (axis + 1) % 3 else request.point[1])
        if value not in self.samples[along]:
            self.samples[along].append(value)
            self.samples[along].sort()


def _surface_edge_length(points: np.ndarray, triangles: np.ndarray, /) -> float:
    """Mean edge length of the discovery contour."""
    edges = np.unique(
        np.sort(
            np.concatenate(
                [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
            ),
            axis=1,
        ),
        axis=0,
    )
    return float(
        np.mean(np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1))
    )


def _face_crease(zero_set: QuotientZeroSet, axis: int, offset: float, /) -> float:
    """Smallest sampled angle (degrees) between the surface and one face plane.

    Samples the periodic face ``u[axis] = offset`` on a regular grid and takes
    the normal's angle to the plane normal at every sample next to a sign
    change of the field (the trace). A face the surface misses scores 90.
    """
    grid = (np.arange(_FACE_SAMPLES, dtype=np.float64) + 0.5) / _FACE_SAMPLES
    first, second = np.meshgrid(grid, grid, indexing="ij")
    points = np.empty((_FACE_SAMPLES, _FACE_SAMPLES, 3), dtype=np.float64)
    points[..., axis] = offset
    points[..., (axis + 1) % 3] = first
    points[..., (axis + 2) % 3] = second
    values = zero_set.values(points.reshape(-1, 3)).reshape(first.shape)
    near = np.zeros(first.shape, dtype=np.bool_)
    for shift in (0, 1):
        change = np.signbit(values) != np.signbit(np.roll(values, -1, axis=shift))
        near |= change | np.roll(change, 1, axis=shift)
    if not np.any(near):
        return 90.0
    gradients = zero_set.gradients(points[near])
    cosine = np.abs(gradients[:, axis]) / np.linalg.norm(gradients, axis=1)
    return float(np.degrees(np.arccos(np.clip(np.max(cosine), 0.0, 1.0))))


def _box_contour(
    surface: PeriodicImplicitSurface, zero_set: QuotientZeroSet, /
) -> QuotientBoxContour:
    """The contour in the fundamental box whose faces cut the surface steepest.

    Each axis takes the candidate offset (``_OFFSETS`` and the published one)
    with the largest sampled crease between the surface and that face plane.
    Sampling only chooses the box; the box's contour is certified by the
    discovery, and the published box is the fallback when the chosen one
    leaves unresolved cells.
    """
    published = np.asarray(surface.reduced_domain[0], dtype=np.float64)
    offset = np.empty(3, dtype=np.float64)
    for axis in range(3):
        candidates = (float(published[axis]), *(float(value) for value in _OFFSETS))
        scores = [_face_crease(zero_set, axis, value) for value in candidates]
        offset[axis] = candidates[int(np.argmax(scores))]
    contour = quotient_box_contour(surface, offset)
    if contour is None:
        contour = quotient_box_contour(surface, published)
    if contour is None:
        raise ValueError("The published quotient box has no contour.")
    return contour


def periodic_box_boundary(
    surface: PeriodicImplicitSurface, maximum_edge_length: float | None, /
) -> PeriodicBoxBoundary:
    """Remeshed contour and refined cell-face caps of a resolved discovery.

    ``maximum_edge_length`` is the reduced length no remeshed contour edge may
    exceed, so the target length is ``maximum_edge_length / _LONG``; ``None``
    targets the box contour's mean edge length.
    """
    zero_set = QuotientZeroSet(surface)
    contour = _box_contour(surface, zero_set)
    length = (
        _surface_edge_length(contour.points, contour.faces)
        if maximum_edge_length is None
        else maximum_edge_length / _LONG
    )
    patch = _Patch(
        zero_set,
        contour.domain,
        contour.points,
        contour.faces,
        contour.root_error,
        length,
    )
    patch.remesh()
    caps = _Caps(patch)
    while True:
        caps_by_axis: list[tuple[_Face, np.ndarray]] = []
        requests: list[_Request] = []
        for axis in range(3):
            outcome = caps.triangulate(axis)
            if isinstance(outcome, list):
                requests.extend(outcome)
            else:
                caps_by_axis.append(outcome)
        if not requests:
            return _assemble(patch, caps_by_axis, contour)
        for request in requests:
            caps.request(request)
        patch.freeze_seams()
        patch.remesh()


def _assemble(
    patch: _Patch, caps: list[tuple[_Face, np.ndarray]], contour: QuotientBoxContour, /
) -> PeriodicBoxBoundary:
    """Lifted points with canonical orbits and windings, contour and cap pairs."""
    low, high = patch.low, patch.high
    rows: dict[bytes, int] = {}
    points: list[np.ndarray] = []

    def row_of(point: np.ndarray, /) -> int:
        key = point.tobytes()
        if key not in rows:
            rows[key] = len(points)
            points.append(point)
        return rows[key]

    surface = [
        [row_of(patch.points[v]) for v in triangle]
        for triangle in patch.triangles
        if triangle is not None
    ]
    error = max(
        patch.errors[v]
        for v in range(len(patch.points))
        if patch.alive[v] and patch.stars[v]
    )
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for face, triangles in caps:
        lower_rows = [row_of(point) for point in face.lifted]
        upper_rows = []
        for point in face.lifted:
            moved = np.array(point, copy=True)
            moved[face.axis] = high[face.axis]
            upper_rows.append(row_of(moved))
        pairs.append(
            (
                np.asarray(lower_rows, dtype=np.int64)[triangles],
                np.asarray(upper_rows, dtype=np.int64)[triangles],
            )
        )
    lifted = np.asarray(points, dtype=np.float64)
    windings = (lifted == high[None, :]).astype(np.int64)
    canonical = np.where(windings == 1, low[None, :], lifted)
    orbit_ids: dict[bytes, int] = {}
    orbits = np.asarray(
        [orbit_ids.setdefault(row.tobytes(), len(orbit_ids)) for row in canonical],
        dtype=np.int64,
    )
    return PeriodicBoxBoundary(
        np.array(contour.domain, copy=True),
        lifted,
        orbits,
        windings,
        np.asarray(surface, dtype=np.int64).reshape(-1, 3),
        tuple(pairs),
        float(error),
        np.array(contour.interior, copy=True),
    )
