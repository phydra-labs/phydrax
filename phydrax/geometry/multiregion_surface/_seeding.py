#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host seeds of labeled multiregion surfaces.

A seed is an immutable host triangulation with region labels, stable global
ids and named vertex sets, from which fixed-capacity topologies and states are
built. Seeds come from explicit converters (3D polyhedral vertex tissues) and
from canonical reference geometries (sphere, standard double bubble, catenoid
band); `MultiRegionSurfaceSeed.subdivided` refines any seed conservatively in
topology (edge midpoints are shared by every incident face, so non-manifold
junction edges stay junction edges).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import final, Literal, TYPE_CHECKING

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import (
    canonical_identifier,
    finite_real_scalar,
    nonnegative_integer,
    positive_finite_float,
    positive_integer,
    unique_identifiers,
)
from ...typing import Dim, HostFloat64, HostInt64, Identifier, parse, Scope
from ._contracts import (
    MultiRegionCoordinateDtype,
    MultiRegionIndexDtype,
    MultiRegionKind,
    MultiRegionSurfaceCapacityPlan,
    MultiRegionSurfaceCounts,
)
from ._state import MultiRegionSurfaceState
from ._topology import _host_incidence, MultiRegionSurfaceTopology


if TYPE_CHECKING:
    from ...applications.cellular_mechanics import VertexTissuePlan


class _SeedVertexDim(Dim, minimum=3):
    """Seed vertices."""


class _SeedFaceDim(Dim, minimum=1):
    """Seed faces."""


_GOLDEN = (1.0 + math.sqrt(5.0)) / 2.0
_ICOSAHEDRON_VERTICES = np.asarray(
    (
        (-1.0, _GOLDEN, 0.0),
        (1.0, _GOLDEN, 0.0),
        (-1.0, -_GOLDEN, 0.0),
        (1.0, -_GOLDEN, 0.0),
        (0.0, -1.0, _GOLDEN),
        (0.0, 1.0, _GOLDEN),
        (0.0, -1.0, -_GOLDEN),
        (0.0, 1.0, -_GOLDEN),
        (_GOLDEN, 0.0, -1.0),
        (_GOLDEN, 0.0, 1.0),
        (-_GOLDEN, 0.0, -1.0),
        (-_GOLDEN, 0.0, 1.0),
    )
)
_ICOSAHEDRON_FACES = np.asarray(
    (
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    )
)  # fmt: skip


def _vertex_sets(
    sets: Mapping[str, Sequence[int]], vertex_ids: np.ndarray, /
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    names = unique_identifiers(tuple(sets), "vertex_sets", allow_empty=True)
    known = set(vertex_ids.tolist())
    result = []
    for name in sorted(names):
        members = tuple(sorted({int(value) for value in sets[name]}))
        if not members or not set(members) <= known:
            raise ValueError(f"vertex set {name!r} must list known global vertex ids.")
        result.append((name, members))
    return tuple(result)


@final
class MultiRegionSurfaceSeed(StrictModule, NonTrainableState):
    """Host triangulation with labels, stable ids and named vertex sets.

    ``face_labels[f] = (left, right)`` index ``region_ids``; normals point out
    of ``left``. ``vertex_sets`` maps names (for example wire rings) to global
    vertex ids.
    """

    __strict_contract__ = True

    positions: HostFloat64[_SeedVertexDim, Literal[3]]
    faces: HostInt64[_SeedFaceDim, Literal[3]]
    face_labels: HostInt64[_SeedFaceDim, Literal[2]]
    vertex_global_ids: HostInt64[_SeedVertexDim]
    face_global_ids: HostInt64[_SeedFaceDim]
    region_ids: tuple[str, ...] = eqx.field(static=True)
    region_kinds: tuple[MultiRegionKind, ...] = eqx.field(static=True)
    vertex_sets: tuple[tuple[str, tuple[int, ...]], ...] = eqx.field(static=True)
    source: Identifier = eqx.field(static=True)
    seed_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        positions: ArrayLike,
        faces: ArrayLike,
        face_labels: ArrayLike,
        region_ids: Sequence[str],
        region_kinds: Sequence[MultiRegionKind],
        /,
        *,
        source: str,
        vertex_global_ids: ArrayLike | None = None,
        face_global_ids: ArrayLike | None = None,
        vertex_sets: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        scope = Scope()
        points = parse(
            np.asarray(positions, dtype=np.float64),
            HostFloat64[_SeedVertexDim, Literal[3]],
            "positions",
            scope=scope,
        )
        rows = parse(
            np.asarray(faces, dtype=np.int64),
            HostInt64[_SeedFaceDim, Literal[3]],
            "faces",
            scope=scope,
        )
        labels = parse(
            np.asarray(face_labels, dtype=np.int64),
            HostInt64[_SeedFaceDim, Literal[2]],
            "face_labels",
            scope=scope,
        )
        ids = unique_identifiers(region_ids, "region_ids")
        kinds = tuple(
            parse(kind, MultiRegionKind, f"region_kinds[{index}]")
            for index, kind in enumerate(region_kinds)
        )
        if len(kinds) != len(ids):
            raise ValueError("region_kinds must give one kind per region id.")
        if not np.all(np.isfinite(points)):
            raise ValueError("Seed positions must be finite.")
        vertex_ids = (
            np.arange(points.shape[0], dtype=np.int64)
            if vertex_global_ids is None
            else np.asarray(vertex_global_ids, dtype=np.int64)
        )
        face_ids = (
            np.arange(rows.shape[0], dtype=np.int64)
            if face_global_ids is None
            else np.asarray(face_global_ids, dtype=np.int64)
        )
        parse(vertex_ids, HostInt64[_SeedVertexDim], "vertex_global_ids", scope=scope)
        parse(face_ids, HostInt64[_SeedFaceDim], "face_global_ids", scope=scope)
        sets = _vertex_sets({} if vertex_sets is None else vertex_sets, vertex_ids)
        source_ = canonical_identifier(source, "source")
        self.positions = points
        self.faces = rows
        self.face_labels = labels
        self.vertex_global_ids = vertex_ids
        self.face_global_ids = face_ids
        self.region_ids = ids
        self.region_kinds = kinds
        self.vertex_sets = sets
        self.source = source_
        self.seed_id = canonical_fingerprint(
            {
                "kind": "multiregion-surface-seed",
                "source": source_,
                "positions": array_tree_fingerprint(points),
                "faces": array_tree_fingerprint(rows),
                "face_labels": array_tree_fingerprint(labels),
                "region_ids": list(ids),
                "region_kinds": list(kinds),
                "vertex_global_ids": array_tree_fingerprint(vertex_ids),
                "face_global_ids": array_tree_fingerprint(face_ids),
                "vertex_sets": [[name, list(members)] for name, members in sets],
            }
        )

    def vertex_set(self, name: str, /) -> tuple[int, ...]:
        """Global vertex ids of one named vertex set."""
        for key, members in self.vertex_sets:
            if key == name:
                return members
        raise ValueError(f"Unknown vertex set {name!r}.")

    def counts(self) -> MultiRegionSurfaceCounts:
        """Exact counts and valence maxima this seed requires."""
        incidence = _host_incidence(self.faces, self.face_labels, self.positions.shape[0])
        return MultiRegionSurfaceCounts(
            vertex=self.positions.shape[0],
            edge=incidence.edges.shape[0],
            face=self.faces.shape[0],
            region=len(self.region_ids),
            region_pair=incidence.region_pairs.shape[0],
            edge_valence=int(np.max(incidence.valence)),
            vertex_region_pairs=int(np.max(incidence.vertex_slot_count)),
        )

    def capacity_plan(
        self,
        *,
        resource_id: str,
        headroom: float = 1.0,
        event_capacity: int = 0,
        coordinate_dtype: MultiRegionCoordinateDtype = "float64",
        index_dtype: MultiRegionIndexDtype = "int32",
    ) -> MultiRegionSurfaceCapacityPlan:
        """Capacity plan fitting this seed with entity-count ``headroom`` >= 1."""
        factor = positive_finite_float(headroom, "headroom")
        if factor < 1.0:
            raise ValueError("headroom must be at least one.")
        counts = self.counts()

        def grown(value: int, /) -> int:
            return max(1, math.ceil(value * factor))

        return MultiRegionSurfaceCapacityPlan(
            vertex_capacity=grown(counts.vertex),
            edge_capacity=grown(counts.edge),
            face_capacity=grown(counts.face),
            region_capacity=grown(counts.region),
            region_pair_capacity=grown(counts.region_pair),
            maximum_edge_valence=counts.edge_valence,
            maximum_vertex_region_pairs=counts.vertex_region_pairs,
            resource_id=resource_id,
            event_capacity=event_capacity,
            coordinate_dtype=coordinate_dtype,
            index_dtype=index_dtype,
        )

    def topology(
        self, plan: MultiRegionSurfaceCapacityPlan, /, *, epoch: int = 0
    ) -> MultiRegionSurfaceTopology:
        """Fixed-capacity topology of this seed under ``plan``."""
        return MultiRegionSurfaceTopology(
            plan,
            self.faces,
            self.face_labels,
            self.region_ids,
            self.region_kinds,
            vertex_count=self.positions.shape[0],
            vertex_global_ids=self.vertex_global_ids,
            face_global_ids=self.face_global_ids,
            epoch=epoch,
        )

    def state(self, topology: MultiRegionSurfaceTopology, /) -> MultiRegionSurfaceState:
        """Capacity-padded state holding the seed positions (zero padding)."""
        if topology.vertex_count != self.positions.shape[0] or not np.array_equal(
            np.asarray(topology.vertex_global_ids[: topology.vertex_count]),
            self.vertex_global_ids,
        ):
            raise ValueError("topology was not built from this seed.")
        padded = np.zeros((topology.vertex_capacity, 3), dtype=np.float64)
        padded[: self.positions.shape[0]] = self.positions
        return MultiRegionSurfaceState(topology, padded)

    def vertex_indices(self, global_ids: Sequence[int], /) -> np.ndarray:
        """Seed vertex slots of stable global vertex ids."""
        lookup = {int(value): index for index, value in enumerate(self.vertex_global_ids)}
        missing = [value for value in global_ids if int(value) not in lookup]
        if missing:
            raise ValueError(f"Unknown global vertex ids {missing[:4]}.")
        return np.asarray([lookup[int(value)] for value in global_ids], dtype=np.int64)

    def subdivided(self, levels: int = 1, /) -> MultiRegionSurfaceSeed:
        """Midpoint subdivision (each face into four) repeated ``levels`` times."""
        count = nonnegative_integer(levels, "levels")
        seed = self
        for _ in range(count):
            seed = _midpoint_subdivision(seed, project=None)
        return seed


def _midpoint_subdivision(
    seed: MultiRegionSurfaceSeed,
    /,
    *,
    project: tuple[np.ndarray, float] | None,
) -> MultiRegionSurfaceSeed:
    faces = seed.faces
    vertex_count = seed.positions.shape[0]
    origin = faces.reshape(-1)
    destination = np.roll(faces, -1, axis=1).reshape(-1)
    keys = np.stack((np.minimum(origin, destination), np.maximum(origin, destination)), 1)
    edges, inverse = np.unique(keys, axis=0, return_inverse=True)
    midpoint = vertex_count + inverse.reshape((-1, 3))
    positions = np.concatenate(
        (seed.positions, 0.5 * (seed.positions[edges[:, 0]] + seed.positions[edges[:, 1]]))
    )
    if project is not None:
        center, radius = project
        offsets = positions - center
        positions = center + radius * offsets / np.linalg.norm(offsets, axis=1)[:, None]
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    ab, bc, ca = midpoint[:, 0], midpoint[:, 1], midpoint[:, 2]
    children = np.stack(
        (
            np.stack((a, ab, ca), axis=1),
            np.stack((ab, b, bc), axis=1),
            np.stack((ca, bc, c), axis=1),
            np.stack((ab, bc, ca), axis=1),
        ),
        axis=1,
    ).reshape((-1, 3))
    labels = np.repeat(seed.face_labels, 4, axis=0)
    first_new = int(np.max(seed.vertex_global_ids)) + 1
    vertex_ids = np.concatenate(
        (seed.vertex_global_ids, first_new + np.arange(edges.shape[0], dtype=np.int64))
    )
    face_ids = (4 * seed.face_global_ids[:, None] + np.arange(4)[None, :]).reshape(-1)
    endpoint_ids = seed.vertex_global_ids[edges]
    sets = {}
    for name, members in seed.vertex_sets:
        member = np.isin(endpoint_ids, np.asarray(members))
        added = vertex_ids[vertex_count:][np.all(member, axis=1)]
        sets[name] = tuple(members) + tuple(int(value) for value in added)
    return MultiRegionSurfaceSeed(
        positions,
        children,
        labels,
        seed.region_ids,
        seed.region_kinds,
        source=f"{seed.source}.subdivided",
        vertex_global_ids=vertex_ids,
        face_global_ids=face_ids,
        vertex_sets=sets,
    )


def _oriented(
    positions: np.ndarray, faces: np.ndarray, direction: np.ndarray, /
) -> np.ndarray:
    """Flip faces whose normal opposes the desired per-face direction."""
    corners = positions[faces]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    flip = np.sum(normal * direction, axis=1) < 0.0
    result = faces.copy()
    result[flip] = result[flip][:, (0, 2, 1)]
    return result


def seed_sphere(
    radius: float,
    /,
    *,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    subdivisions: int = 2,
    region_id: str = "bubble",
    ambient_id: str = "ambient",
) -> MultiRegionSurfaceSeed:
    """Icosphere bubble (finite ``region_id``) in the ``ambient_id`` boundary label."""
    radius_ = positive_finite_float(radius, "radius")
    levels = nonnegative_integer(subdivisions, "subdivisions")
    center_ = np.asarray([finite_real_scalar(value, "center") for value in center])
    if center_.shape != (3,):
        raise ValueError("center must have three coordinates.")
    unit = _ICOSAHEDRON_VERTICES / np.linalg.norm(_ICOSAHEDRON_VERTICES, axis=1)[:, None]
    seed = MultiRegionSurfaceSeed(
        center_ + radius_ * unit,
        _ICOSAHEDRON_FACES,
        np.zeros((20, 2), dtype=np.int64) + np.asarray((0, 1), dtype=np.int64),
        (region_id, ambient_id),
        ("finite", "boundary"),
        source="icosphere",
    )
    for _ in range(levels):
        seed = _midpoint_subdivision(seed, project=(center_, radius_))
    faces = _oriented(
        seed.positions,
        seed.faces,
        np.mean(seed.positions[seed.faces], axis=1) - center_,
    )
    return MultiRegionSurfaceSeed(
        seed.positions,
        faces,
        seed.face_labels,
        seed.region_ids,
        seed.region_kinds,
        source="icosphere",
        vertex_global_ids=seed.vertex_global_ids,
        face_global_ids=seed.face_global_ids,
    )


def _strip(inner: np.ndarray, outer: np.ndarray, /) -> np.ndarray:
    """Triangulate between two closed rings by merging their angular order.

    ``inner``/``outer`` hold ``(vertex_index, angle)`` rows sorted by angle in
    ``[0, 2 pi)``; the strip is closed by wrapping both rings once.
    """
    inner_count, outer_count = inner.shape[0], outer.shape[0]
    triangles = []
    i = j = 0
    while i < inner_count or j < outer_count:
        a = inner[i % inner_count]
        b = outer[j % outer_count]
        next_inner = inner[(i + 1) % inner_count, 1] + 2.0 * np.pi * ((i + 1) // inner_count)
        next_outer = outer[(j + 1) % outer_count, 1] + 2.0 * np.pi * ((j + 1) // outer_count)
        advance_inner = j >= outer_count or (i < inner_count and next_inner <= next_outer)
        if advance_inner:
            triangles.append((a[0], b[0], inner[(i + 1) % inner_count, 0]))
            i += 1
        else:
            triangles.append((a[0], b[0], outer[(j + 1) % outer_count, 0]))
            j += 1
    return np.asarray(triangles, dtype=np.int64)


def _ring_rows(
    start: int, count: int, offset: float, /
) -> np.ndarray:
    angles = offset + 2.0 * np.pi * np.arange(count) / count
    return np.stack((start + np.arange(count), np.mod(angles, 2.0 * np.pi)), axis=1)


def _polar_cap(
    ring_indices: np.ndarray,
    ring_angles: np.ndarray,
    point: Callable[[int, np.ndarray], np.ndarray],
    ring_counts: Sequence[int],
    first_index: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Apex fan plus merged strips from the apex out to a shared outer ring.

    ``point(k, angles)`` returns positions of interior ring ``k`` (``k = 0`` is
    the apex). Returns new vertex positions and faces (orientation unfixed).
    """
    positions = [point(0, np.zeros((1,)))]
    faces = []
    apex = first_index
    next_index = first_index + 1
    previous = None
    for level, requested in enumerate(ring_counts, start=1):
        # Odd interior ring counts avoid exactly antipodal (collinear-through-
        # apex) vertex pairs, which filtered predicates cannot certify.
        count = requested | 1
        offset = (level % 2) * np.pi / count
        rows = _ring_rows(next_index, count, offset)
        positions.append(point(level, rows[:, 1]))
        if previous is None:
            faces.append(
                np.stack(
                    (np.full(count, apex), rows[:, 0], np.roll(rows[:, 0], -1)), axis=1
                ).astype(np.int64)
            )
        else:
            faces.append(_strip(previous, rows))
        previous = rows
        next_index += count
    outer = np.stack((ring_indices, ring_angles), axis=1)
    order = np.argsort(outer[:, 1], kind="stable")
    if previous is None:
        faces.append(
            np.stack(
                (
                    np.full(outer.shape[0], apex),
                    outer[order, 0],
                    np.roll(outer[order, 0], -1),
                ),
                axis=1,
            ).astype(np.int64)
        )
    else:
        faces.append(_strip(previous, outer[order]))
    return np.concatenate(positions), np.concatenate(faces).astype(np.int64)


def _spherical_cap(
    center: np.ndarray,
    radius: float,
    axis_sign: float,
    polar_limit: float,
    spacing: float,
    ring_indices: np.ndarray,
    ring_angles: np.ndarray,
    first_index: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap of the sphere ``(center, radius)`` from its apex on ``axis_sign * z``."""
    levels = max(1, round(radius * polar_limit / spacing))
    polar = polar_limit * np.arange(1, levels) / levels
    counts = [max(6, round(2.0 * np.pi * radius * np.sin(angle) / spacing)) for angle in polar]

    def point(level: int, angles: np.ndarray, /) -> np.ndarray:
        psi = 0.0 if level == 0 else polar[level - 1]
        return center + radius * np.stack(
            (
                np.sin(psi) * np.cos(angles),
                np.sin(psi) * np.sin(angles),
                np.full(angles.shape, axis_sign * np.cos(psi)),
            ),
            axis=1,
        )

    return _polar_cap(ring_indices, ring_angles, point, counts, first_index)


def _flat_disk(
    ring_radius: float,
    height: float,
    spacing: float,
    ring_indices: np.ndarray,
    ring_angles: np.ndarray,
    first_index: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    levels = max(1, round(ring_radius / spacing))
    radii = ring_radius * np.arange(1, levels) / levels
    counts = [max(6, round(2.0 * np.pi * value / spacing)) for value in radii]

    def point(level: int, angles: np.ndarray, /) -> np.ndarray:
        r = 0.0 if level == 0 else radii[level - 1]
        return np.stack(
            (r * np.cos(angles), r * np.sin(angles), np.full(angles.shape, height)), axis=1
        )

    return _polar_cap(ring_indices, ring_angles, point, counts, first_index)


def seed_double_bubble(
    radius_first: float,
    radius_second: float,
    /,
    *,
    ring_points: int = 24,
    region_ids: tuple[str, str] = ("bubble-1", "bubble-2"),
    ambient_id: str = "ambient",
) -> MultiRegionSurfaceSeed:
    """Standard equal-tension double bubble with outer radii ``R1``, ``R2``.

    The outer spheres meet on a circle in the plane ``z = 0`` with center
    distance ``d^2 = R1^2 + R2^2 - R1 R2`` (films at 120 degrees); the separating
    film is the spherical cap of radius ``R1 R2 / |R1 - R2|`` (flat for equal
    radii) bulging into the larger bubble. Face labels are
    ``(bubble-1, ambient)``, ``(bubble-2, ambient)`` and ``(bubble-1, bubble-2)``.
    The vertex set ``"junction"`` is the shared triple ring.
    """
    first = positive_finite_float(radius_first, "radius_first")
    second = positive_finite_float(radius_second, "radius_second")
    count = positive_integer(ring_points, "ring_points")
    if count < 6:
        raise ValueError("ring_points must be at least six.")
    distance = math.sqrt(first * first + second * second - first * second)
    offset = (distance * distance + first * first - second * second) / (2.0 * distance)
    ring_radius = math.sqrt(first * first - offset * offset)
    spacing = 2.0 * math.pi * ring_radius / count
    # An offset incommensurate with pi keeps junction vertices off the rational
    # angles of the interior rings (no exactly collinear vertex triples).
    ring_angles = 2.0 * np.pi * (np.arange(count) + (math.sqrt(5.0) - 2.0)) / count
    ring_indices = np.arange(count, dtype=np.int64)
    ring = np.stack(
        (ring_radius * np.cos(ring_angles), ring_radius * np.sin(ring_angles), np.zeros(count)),
        axis=1,
    )
    center_first = np.asarray((0.0, 0.0, -offset))
    center_second = np.asarray((0.0, 0.0, distance - offset))
    cap_first, faces_first = _spherical_cap(
        center_first,
        first,
        -1.0,
        math.acos(-offset / first),
        spacing,
        ring_indices,
        ring_angles,
        count,
    )
    start_second = count + cap_first.shape[0]
    cap_second, faces_second = _spherical_cap(
        center_second,
        second,
        1.0,
        math.acos(-(distance - offset) / second),
        spacing,
        ring_indices,
        ring_angles,
        start_second,
    )
    start_wall = start_second + cap_second.shape[0]
    if math.isclose(first, second, rel_tol=1e-12):
        wall, faces_wall = _flat_disk(
            ring_radius, 0.0, spacing, ring_indices, ring_angles, start_wall
        )
    else:
        wall_radius = first * second / abs(first - second)
        wall_sign = -1.0 if first > second else 1.0
        depth = math.sqrt(wall_radius * wall_radius - ring_radius * ring_radius)
        wall_center = np.asarray((0.0, 0.0, -wall_sign * depth))
        wall, faces_wall = _spherical_cap(
            wall_center,
            wall_radius,
            wall_sign,
            math.asin(ring_radius / wall_radius),
            spacing,
            ring_indices,
            ring_angles,
            start_wall,
        )
    positions = np.concatenate((ring, cap_first, cap_second, wall))
    faces_first = _oriented(
        positions, faces_first, np.mean(positions[faces_first], axis=1) - center_first
    )
    faces_second = _oriented(
        positions, faces_second, np.mean(positions[faces_second], axis=1) - center_second
    )
    faces_wall = _oriented(
        positions, faces_wall, np.broadcast_to((0.0, 0.0, 1.0), faces_wall.shape)
    )
    labels = np.concatenate(
        (
            np.broadcast_to((0, 2), faces_first.shape[:1] + (2,)),
            np.broadcast_to((1, 2), faces_second.shape[:1] + (2,)),
            np.broadcast_to((0, 1), faces_wall.shape[:1] + (2,)),
        )
    )
    return MultiRegionSurfaceSeed(
        positions,
        np.concatenate((faces_first, faces_second, faces_wall)),
        labels,
        (region_ids[0], region_ids[1], ambient_id),
        ("finite", "finite", "boundary"),
        source="standard-double-bubble",
        vertex_sets={"junction": tuple(range(count))},
    )


def seed_catenoid(
    ring_radius: float,
    half_separation: float,
    /,
    *,
    ring_points: int = 32,
    rows: int = 12,
    neck_radius: float | None = None,
    inner_id: str = "core",
    outer_id: str = "ambient",
) -> MultiRegionSurfaceSeed:
    """Film band spanning coaxial rings of radius ``R`` at ``z = +/- d``.

    Both sides are boundary labels at the ambient pressure (no volume
    constraint). The initial profile is the cylinder ``r = R`` or, with
    ``neck_radius``, the parabola from ``R`` at the rings to ``neck_radius`` at
    ``z = 0``. Vertex sets ``"ring-lower"`` and ``"ring-upper"`` are the wires.
    """
    radius = positive_finite_float(ring_radius, "ring_radius")
    half = positive_finite_float(half_separation, "half_separation")
    count = positive_integer(ring_points, "ring_points")
    levels = positive_integer(rows, "rows")
    if count < 6 or levels < 2:
        raise ValueError("A catenoid band needs ring_points >= 6 and rows >= 2.")
    neck = radius if neck_radius is None else positive_finite_float(neck_radius, "neck_radius")
    heights = np.linspace(-half, half, levels + 1)
    profile = radius - (radius - neck) * (1.0 - (heights / half) ** 2)
    angles = 2.0 * np.pi * np.arange(count) / count
    shifts = np.where(np.arange(levels + 1) % 2 == 0, 0.0, np.pi / count)
    theta = angles[None, :] + shifts[:, None]
    positions = np.stack(
        (
            profile[:, None] * np.cos(theta),
            profile[:, None] * np.sin(theta),
            np.broadcast_to(heights[:, None], theta.shape),
        ),
        axis=2,
    ).reshape((-1, 3))
    index = np.arange((levels + 1) * count).reshape((levels + 1, count))
    lower, upper = index[:-1], index[1:]
    lower_next, upper_next = np.roll(lower, -1, axis=1), np.roll(upper, -1, axis=1)
    even = (np.arange(levels) % 2 == 0)[:, None, None]
    first = np.where(
        even,
        np.stack((lower, lower_next, upper), axis=2),
        np.stack((lower, lower_next, upper_next), axis=2),
    )
    second = np.where(
        even,
        np.stack((lower_next, upper_next, upper), axis=2),
        np.stack((lower, upper_next, upper), axis=2),
    )
    faces = np.concatenate((first.reshape((-1, 3)), second.reshape((-1, 3))))
    centroid = np.mean(positions[faces], axis=1)
    outward = centroid * np.asarray((1.0, 1.0, 0.0))
    faces = _oriented(positions, faces, outward)
    return MultiRegionSurfaceSeed(
        positions,
        faces,
        np.broadcast_to((0, 1), faces.shape[:1] + (2,)),
        (inner_id, outer_id),
        ("boundary", "boundary"),
        source="catenoid-band",
        vertex_sets={
            "ring-lower": tuple(int(value) for value in index[0]),
            "ring-upper": tuple(int(value) for value in index[-1]),
        },
    )


def _tissue_face_owners(
    cell_faces: np.ndarray, cell_signs: np.ndarray, face_count: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Owner cells and orientation signs of every face slot (-1 / 0 when absent)."""
    owners = np.full((face_count, 2), -1, dtype=np.int64)
    signs = np.zeros((face_count, 2), dtype=np.int64)
    cells, columns = np.nonzero(cell_faces >= 0)
    faces = cell_faces[cells, columns]
    order = np.lexsort((cells, faces))
    faces, cells, columns = faces[order], cells[order], columns[order]
    rank = np.zeros_like(faces)
    rank[1:] = np.where(faces[1:] == faces[:-1], 1, 0)
    owners[faces, rank] = cells
    signs[faces, rank] = cell_signs[cells, columns]
    return owners, signs


def seed_from_vertex_tissue(
    plan: VertexTissuePlan,
    positions: ArrayLike,
    /,
    *,
    ambient_id: str = "ambient",
    region_prefix: str = "cell-",
) -> MultiRegionSurfaceSeed:
    """Convert an oriented 3D polyhedral vertex tissue into a multiregion seed.

    Every active polygon face is fan-triangulated from its first loop vertex.
    A face owned by two cells separates them; a face owned by one cell
    separates it from ``ambient_id``. Labels are oriented so the triangle
    normal points out of the owner whose ``cell_face_orientations`` entry is
    ``+1`` (the loop is reversed when the only owner has ``-1``). Regions are
    ``f"{region_prefix}{cell_id}"`` with the tissue's stable cell ids; vertex
    global ids are the tissue vertex ids and triangle ``k`` of tissue face
    ``F`` gets global id ``F * max_triangles + k``. The tissue is not modified;
    its constitutive data are not converted.
    """
    from ...applications.cellular_mechanics import VertexTissuePlan

    if not isinstance(plan, VertexTissuePlan):
        raise TypeError("plan must be a VertexTissuePlan.")
    if plan.dimension != 3:
        raise ValueError("Only 3D polyhedral vertex tissues convert to multiregion seeds.")
    prefix = canonical_identifier(region_prefix, "region_prefix")
    coordinates = np.asarray(positions, dtype=np.float64)
    if coordinates.shape != (plan.vertex_capacity, 3):
        raise ValueError("positions must have shape (vertex_capacity, 3).")
    face_active = np.asarray(plan.face_active)
    cell_active = np.asarray(plan.cell_active)
    vertex_active = np.asarray(plan.vertex_active)
    loops = np.asarray(plan.face_vertex_indices, dtype=np.int64)
    owners, owner_signs = _tissue_face_owners(
        np.asarray(plan.cell_face_indices, dtype=np.int64),
        np.asarray(plan.cell_face_orientations, dtype=np.int64),
        loops.shape[0],
    )
    cell_ids = np.asarray(plan.cell_ids, dtype=np.int64)
    active_cells = np.flatnonzero(cell_active)
    region_ids = tuple(f"{prefix}{int(cell_ids[cell])}" for cell in active_cells) + (
        canonical_identifier(ambient_id, "ambient_id"),
    )
    region_of_cell = np.full((cell_active.size,), -1, dtype=np.int64)
    region_of_cell[active_cells] = np.arange(active_cells.size)
    ambient = active_cells.size
    active_vertices = np.flatnonzero(vertex_active)
    compact = np.full((vertex_active.size,), -1, dtype=np.int64)
    compact[active_vertices] = np.arange(active_vertices.size)
    width = loops.shape[1]
    triangles, labels, face_ids = [], [], []
    tissue_face_ids = np.asarray(plan.face_ids, dtype=np.int64)
    for face in np.flatnonzero(face_active):
        loop = loops[face][loops[face] >= 0]
        first_sign = owner_signs[face, 0]
        second_owner = owners[face, 1]
        # The stored loop is outward for the owner with orientation +1, which
        # is therefore the left label; a lone inward owner reverses the loop.
        if second_owner >= 0:
            left_cell = owners[face, 0] if first_sign > 0 else second_owner
            right_region = region_of_cell[
                second_owner if first_sign > 0 else owners[face, 0]
            ]
        else:
            left_cell = owners[face, 0]
            right_region = ambient
            if first_sign < 0:
                loop = loop[::-1]
        fan = np.stack(
            (np.full(loop.size - 2, loop[0]), loop[1:-1], loop[2:]), axis=1
        )
        triangles.append(compact[fan])
        labels.append(
            np.broadcast_to((region_of_cell[left_cell], right_region), (loop.size - 2, 2))
        )
        face_ids.append(tissue_face_ids[face] * (width - 2) + np.arange(loop.size - 2))
    return MultiRegionSurfaceSeed(
        coordinates[active_vertices],
        np.concatenate(triangles),
        np.concatenate(labels),
        region_ids,
        ("finite",) * active_cells.size + ("boundary",),
        source="vertex-tissue",
        vertex_global_ids=np.asarray(plan.vertex_ids, dtype=np.int64)[active_vertices],
        face_global_ids=np.concatenate(face_ids),
    )


__all__ = [
    "MultiRegionSurfaceSeed",
    "seed_catenoid",
    "seed_double_bubble",
    "seed_from_vertex_tissue",
    "seed_sphere",
]
