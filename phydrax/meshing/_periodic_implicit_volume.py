#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Periodic tetrahedral fill of a quotient-discovered trigonometric region.

The fundamental cell's region is closed by its lifted quotient contour and by
planar caps on the cell faces. Each cap is the region's trace on a lower cell
face, tiled by the faces of the certified discovery cover and cut exactly by
the contour edges lying in that face; the upper face carries the identical
triangulation moved by the exact lattice action. The resulting piecewise-linear
complex is filled by the native fixed-boundary PLC route, which keeps every
source vertex and facet, so paired cell faces stay identical. Lifted vertex
orbits then bind the canonical ``PeriodicMeshTopology``: cap facets become
interior quotient faces and only the contour bounds the quotient region.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np

from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_mesh import CellBlock, CellMesh
from ..discretization._periodic_topology import PeriodicMeshTopology
from ..geometry.implicit._periodic_discovery import (
    lift_reduced_points,
    PeriodicImplicitSurface,
)
from ..typing import checked
from ._canonical import certify_cell_mesh
from ._contracts import CellFamilyPolicy, CellMeshingTarget, VolumeFillStrategy
from ._controls import RegionSeed
from ._organization import RegionRole
from ._plc_source import PiecewiseLinearComplex
from ._result import CellMeshingResult
from ._scope import MeshingEntityKind, MeshingScope
from ._sizing import SizeControlStrength, UniformSizeControl
from ._volume_generation import NativeVolumeSchedule


type _Vertex = tuple[str, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class _Bank:
    """Lifted PLC vertices: reduced points, orbit keys and windings."""

    points: list[np.ndarray]
    orbits: list[tuple[object, ...]]
    windings: list[tuple[int, int, int]]
    index: dict[_Vertex, int]

    def add(
        self,
        key: _Vertex,
        point: np.ndarray,
        orbit: tuple[object, ...],
        winding: tuple[int, int, int],
        /,
    ) -> int:
        row = self.index.get(key)
        if row is None:
            row = len(self.points)
            self.index[key] = row
            self.points.append(np.asarray(point, dtype=np.float64))
            self.orbits.append(orbit)
            self.windings.append(winding)
        return row


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray, /) -> int:
    """Exact sign of the 2D orientation of three binary64 points."""
    ax, ay, bx, by, cx, cy = (Fraction(float(value)) for value in (*a, *b, *c))
    value = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    return (value > 0) - (value < 0)


@dataclass(frozen=True, slots=True)
class _Item:
    """One boundary-cycle entry of a cap tile: a grid corner or a contour root."""

    corner: tuple[int, int] | None
    vertex: int


def _triangulate_convex(
    loop: list[int], plane: Mapping[int, np.ndarray], /
) -> list[tuple[int, int, int]]:
    """Triangulate a counter-clockwise convex loop that may hold collinear vertices.

    Only strictly convex ears are clipped, and never one whose removal would
    leave the remaining loop without a strictly convex vertex, so every
    triangle is nondegenerate under exact orientation.
    """
    remaining = list(loop)
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        for position in range(len(remaining)):
            previous = remaining[position - 1]
            current = remaining[position]
            following = remaining[(position + 1) % len(remaining)]
            if _orientation(plane[previous], plane[current], plane[following]) <= 0:
                continue
            rest = remaining[:position] + remaining[position + 1 :]
            if not any(
                _orientation(
                    plane[rest[index - 1]],
                    plane[rest[index]],
                    plane[rest[(index + 1) % len(rest)]],
                )
                > 0
                for index in range(len(rest))
            ):
                continue
            triangles.append((previous, current, following))
            remaining = rest
            break
        else:
            raise ValueError("A cap polygon is degenerate under exact orientation.")
    if _orientation(plane[remaining[0]], plane[remaining[1]], plane[remaining[2]]) <= 0:
        raise ValueError("A cap polygon is degenerate under exact orientation.")
    triangles.append((remaining[0], remaining[1], remaining[2]))
    return triangles


def _face_caps(
    surface: PeriodicImplicitSurface,
    bank: _Bank,
    axis: int,
    surface_rows: np.ndarray,
    root_vertex: dict[tuple[int, tuple[int, int, int]], int],
    cut_pairs: set[tuple[int, int]],
    /,
) -> list[tuple[int, int, int]]:
    """Counter-clockwise (normal ``+e_axis``) cap triangles of the lower face.

    The face is tiled by the target-level grid squares, never by coarse cover
    leaves: the same squares tile every identified face and box edge, so the
    paired cell faces share one vertex set without hanging vertices. A square
    inside a coarse leaf takes that leaf's certified sign.
    """
    span = 1 << surface.target_level
    offset = surface.reduced_domain[0]
    first, second = (axis + 1) % 3, (axis + 2) % 3
    cover = surface.cover
    boxes = np.asarray(cover.boxes, dtype=np.float64)
    on_face = np.flatnonzero(boxes[:, 0, axis] == offset[axis])
    ids = np.rint((boxes[on_face] - offset[None, None, :]) * span).astype(np.int64)
    # Square classes: -1 inside, +1 outside, 0 band (a target-level leaf).
    classes = np.full((span, span), 2, dtype=np.int8)
    value_lower = np.asarray(cover.value_lower)[on_face]
    value_upper = np.asarray(cover.value_upper)[on_face]
    for row in range(on_face.size):
        low, high = ids[row, 0], ids[row, 1]
        classes[low[first] : high[first], low[second] : high[second]] = (
            -1 if value_upper[row] < 0.0 else 1 if value_lower[row] > 0.0 else 0
        )
    if np.any(classes == 2):
        raise ValueError("The quotient cover does not tile a cell face.")
    signs = {
        (int(key[first]), int(key[second])): int(sign)
        for key, sign in zip(
            surface.scaffold_corner_ids.tolist(),
            surface.scaffold_corner_signs.tolist(),
            strict=True,
        )
        if key[axis] == 0
    }

    def corner_vertex(point: tuple[int, int], /) -> int:
        grid = [0, 0, 0]
        grid[first], grid[second] = point
        coordinates = offset + np.asarray(grid, dtype=np.float64) / span
        canonical = tuple(value % span for value in grid)
        winding = (grid[0] // span, grid[1] // span, grid[2] // span)
        return bank.add(
            ("corner", tuple(grid)), coordinates, ("corner", canonical), winding
        )

    # Roots on the face edges, keyed by the edge's (a, b) endpoint pair.
    face_roots: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    for row in surface_rows.tolist():
        edge = surface.scaffold_root_edge_ids[row]
        if edge[0][axis] != 0 or edge[1][axis] != 0:
            continue
        a = (int(edge[0][first]), int(edge[0][second]))
        b = (int(edge[1][first]), int(edge[1][second]))
        face_roots[(min(a, b), max(a, b))] = root_vertex[(row, (0, 0, 0))]

    plane: dict[int, np.ndarray] = {}
    triangles: list[tuple[int, int, int]] = []
    for a, b in np.argwhere(classes <= 0).tolist():
        corners = ((a, b), (a + 1, b), (a + 1, b + 1), (a, b + 1))
        cycle: list[_Item] = []
        for index, point in enumerate(corners):
            following = corners[(index + 1) % 4]
            cycle.append(_Item(point, -1))
            root = face_roots.get((min(point, following), max(point, following)))
            if root is not None:
                cycle.append(_Item(None, root))
        item_loops = (
            [cycle] if classes[a, b] < 0 else _band_loops(cycle, signs, cut_pairs)
        )
        for items in item_loops:
            # Only corners on an inside loop become cap vertices.
            loop = [
                item.vertex if item.corner is None else corner_vertex(item.corner)
                for item in items
            ]
            for vertex in loop:
                if vertex not in plane:
                    point = bank.points[vertex]
                    plane[vertex] = np.asarray((point[first], point[second]))
            triangles.extend(_triangulate_convex(loop, plane))
    return triangles


def _band_loops(
    cycle: list[_Item],
    signs: dict[tuple[int, int], int],
    cut_pairs: set[tuple[int, int]],
    /,
) -> list[list[_Item]]:
    """Inside loops of a band tile, cut along the contour edges in its face."""

    def arc_signs(items: list[_Item], /) -> set[int]:
        return {signs[item.corner] for item in items if item.corner is not None}

    roots = [index for index, item in enumerate(cycle) if item.corner is None]
    if not roots:
        corner_signs = arc_signs(cycle)
        if corner_signs == {-1}:
            return [cycle]
        if corner_signs == {1}:
            return []
        raise ValueError("A cap tile without roots must have one corner sign.")
    count = len(cycle)
    # Arcs run from one root to the next along the boundary cycle.
    arcs: dict[int, tuple[list[_Item], int]] = {}
    for position, start in enumerate(roots):
        stop = roots[(position + 1) % len(roots)]
        items = [
            cycle[(start + step) % count]
            for step in range(1, (stop - start) % count or count)
        ]
        arcs[start] = (items, stop)
    partner: dict[int, int] = {}
    for a in roots:
        for b in roots:
            first, second = cycle[a].vertex, cycle[b].vertex
            if a != b and ((first, second) in cut_pairs or (second, first) in cut_pairs):
                partner[a] = b
    if len(partner) != len(roots):
        raise ValueError("Cap tile roots must pair with contour edges in the face.")
    loops: list[list[_Item]] = []
    visited: set[int] = set()
    for start in roots:
        signs_here = arc_signs(arcs[start][0])
        if signs_here not in ({-1}, {1}):
            raise ValueError("A cap arc between roots must have one corner sign.")
        if signs_here != {-1} or start in visited:
            continue
        loop: list[_Item] = []
        current = start
        while current not in visited:
            visited.add(current)
            items, stop = arcs[current]
            loop.append(cycle[current])
            loop.extend(items)
            loop.append(cycle[stop])
            current = partner[stop]
        if current != start:
            raise ValueError("A cap loop does not close through its contour cuts.")
        loops.append(loop)
    return loops


@final
class PeriodicImplicitVolume(StrictModule, NonTrainableState):
    """Certified periodic tetrahedral mesh of a quotient-discovered region.

    ``result`` is the certified lifted mesh carrying the canonical
    ``PeriodicMeshTopology``. ``quotient_region_components`` counts connected
    components of the quotient tetrahedra (face adjacency through paired cell
    faces). ``measure`` is the mesh volume of the fundamental cell's region.
    ``construction`` is the native fixed-boundary fill of the lifted cell with
    its compliance report (including unmet soft shape targets).
    """

    surface: PeriodicImplicitSurface
    result: CellMeshingResult
    construction: CellMeshingResult
    cap_triangle_count: int = eqx.field(static=True)
    quotient_region_components: int = eqx.field(static=True)
    measure: float = eqx.field(static=True)

    def __init__(
        self,
        surface: PeriodicImplicitSurface,
        result: CellMeshingResult,
        construction: CellMeshingResult,
        /,
        *,
        cap_triangle_count: int,
        quotient_region_components: int,
        measure: float,
    ) -> None:
        self.surface = surface
        self.result = result
        self.construction = construction
        self.cap_triangle_count = int(cap_triangle_count)
        self.quotient_region_components = int(quotient_region_components)
        self.measure = float(measure)

    @property
    def mesh(self) -> CellMesh:
        return self.result.mesh


def _quotient_components(mesh: CellMesh, topology: PeriodicMeshTopology, /) -> int:
    """Connected components of quotient tetrahedra under shared quotient faces."""
    tetrahedra = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    orbits = np.asarray(topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    parent = list(range(tetrahedra.shape[0]))

    def find(item: int, /) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    owner: dict[tuple[object, ...], int] = {}
    for row, cell in enumerate(tetrahedra.tolist()):
        for omitted in range(4):
            face = [cell[index] for index in range(4) if index != omitted]
            base = min(face, key=lambda vertex: (orbits[vertex], tuple(shifts[vertex])))
            key = tuple(
                sorted(
                    (
                        int(orbits[vertex]),
                        *(int(value) for value in shifts[vertex] - shifts[base]),
                    )
                    for vertex in face
                )
            )
            other = owner.setdefault(key, row)
            if other != row:
                a, b = find(other), find(row)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return len({find(row) for row in range(tetrahedra.shape[0])})


@checked
def mesh_periodic_trigonometric_region(
    surface: PeriodicImplicitSurface,
    /,
    *,
    maximum_edge_length: float | None = None,
    volume_schedule: NativeVolumeSchedule | None = None,
    coordinate_contract: SpatialCoordinateContract | None = None,
    source_revision: str = "revision-1",
) -> PeriodicImplicitVolume:
    """Fill the quotient region of a resolved periodic discovery with tetrahedra.

    ``maximum_edge_length`` is an optional hard uniform size target on the
    contour scope. The native route keeps every contour and cap vertex and
    facet (``boundary="fixed"``), so the paired cell-face triangulations stay
    identical; the periodic topology then identifies them exactly. The fixed
    contour is the discovery's lattice contour, so the default schedule makes
    tetrahedral shape targets soft: unmet shapes are reported in the result's
    compliance (``soft_shape_target_unmet_count``), never silently passed.
    """
    from ._contracts import VolumeMeshingSpec
    from .providers._native import NativeMeshingProvider
    from .providers._native_options import NativeMeshingOptions
    from .providers._native_sources import NativePlcSource

    if not surface.resolved:
        raise ValueError(
            "Only a resolved quotient discovery can be meshed; see its evidence."
        )
    span = 1 << surface.target_level
    bank = _Bank([], [], [], {})
    root_vertex: dict[tuple[int, tuple[int, int, int]], int] = {}
    surface_vertices: list[int] = []
    for vertex in range(surface.reduced_points.shape[0]):
        winding = tuple(int(value) for value in surface.vertex_windings[vertex])
        row = bank.add(
            ("root", (int(surface.vertex_root_rows[vertex]),)),
            surface.reduced_points[vertex],
            ("root", int(surface.vertex_orbits[vertex])),
            (winding[0], winding[1], winding[2]),
        )
        surface_vertices.append(row)
    by_orbit = {(bank.orbits[row], bank.windings[row]): row for row in surface_vertices}
    for vertex, row in enumerate(surface_vertices):
        root_vertex[(int(surface.vertex_root_rows[vertex]), (0, 0, 0))] = row
    surface_triangles = [
        (
            surface_vertices[int(face[0])],
            surface_vertices[int(face[1])],
            surface_vertices[int(face[2])],
        )
        for face in surface.faces
    ]
    cut_pairs: set[tuple[int, int]] = set()
    for a, b, c in surface_triangles:
        cut_pairs.update(((a, b), (b, c), (c, a)))
    lower_root_rows = np.asarray(surface.vertex_root_rows, dtype=np.int64)
    polygons: list[tuple[int, int, int]] = list(surface_triangles)
    facets: list[int] = [0] * len(surface_triangles)
    segments: list[tuple[int, int]] = []
    cap_count = 0
    for axis in range(3):
        lower = _face_caps(surface, bank, axis, lower_root_rows, root_vertex, cut_pairs)
        step = [0, 0, 0]
        step[axis] = 1

        def translate(row: int, /) -> int:
            orbit, winding = bank.orbits[row], bank.windings[row]
            moved = (winding[0] + step[0], winding[1] + step[1], winding[2] + step[2])
            if orbit[0] == "root":
                partner = by_orbit.get((orbit, moved))
                if partner is None:
                    raise ValueError("A cap root has no translate on the paired face.")
                return partner
            grid = [0, 0, 0]
            point = np.rint((bank.points[row] - surface.reduced_domain[0]) * span)
            for index in range(3):
                grid[index] = int(point[index]) + span * step[index]
            canonical = orbit[1]
            return bank.add(
                ("corner", tuple(grid)),
                surface.reduced_domain[0] + np.asarray(grid, dtype=np.float64) / span,
                ("corner", canonical),
                moved,
            )

        # The lower face's outward normal is -e_axis; its translate keeps the
        # counter-clockwise (+e_axis) orientation as the upper face's outward side.
        for a, b, c in lower:
            polygons.append((a, c, b))
            facets.append(1 + 2 * axis)
            polygons.append((translate(a), translate(b), translate(c)))
            facets.append(2 + 2 * axis)
        cap_count += 2 * len(lower)
        # Native source preparation merges the coplanar triangles of one cap
        # facet and drops their shared edges, after which recovery may pick
        # either diagonal of a cocircular grid square on each face. Authoring
        # every interior cap edge, and its exact translate, as a segment keeps
        # both triangulations and so the lattice pairing of the cell faces.
        uses: dict[tuple[int, int], int] = {}
        for a, b, c in lower:
            for first, second in ((a, b), (b, c), (c, a)):
                key = (min(first, second), max(first, second))
                uses[key] = uses.get(key, 0) + 1
        for (first, second), count in uses.items():
            if count == 2:
                segments.append((first, second))
                segments.append((translate(first), translate(second)))
    points = np.asarray(bank.points, dtype=np.float64)
    orbit_ids: dict[tuple[object, ...], int] = {}
    orbits = np.asarray(
        [orbit_ids.setdefault(orbit, len(orbit_ids)) for orbit in bank.orbits],
        dtype=np.int64,
    )
    windings = np.asarray(bank.windings, dtype=np.int64)
    cell = surface.cell
    coordinates, representatives, windings = lift_reduced_points(
        cell, points, orbits, windings
    )
    loops = np.asarray(polygons, dtype=np.int64)
    if np.linalg.det(np.asarray(cell.vectors, dtype=np.float64)) < 0.0:
        loops = loops[:, ::-1].copy()
    facet_rows = np.asarray(facets, dtype=np.int64)
    source_id = f"{surface.source_id}-periodic-region"
    complex_ = PiecewiseLinearComplex(
        coordinates,
        loops,
        facet_rows,
        np.tile(np.asarray((-1, 0), dtype=np.int64), (7, 1)),
        ("region",),
        segments=np.asarray(segments, dtype=np.int64).reshape(-1, 2),
        vertex_ids=tuple(f"{source_id}-vertex-{row}" for row in range(points.shape[0])),
        vertex_occurrence_paths=((),) * points.shape[0],
        facet_ids=(
            f"{source_id}-contour",
            *(
                f"{source_id}-cell-face-{axis}-{side}"
                for axis in range(3)
                for side in ("lower", "upper")
            ),
        ),
        facet_occurrence_paths=((),) * 7,
        segment_ids=tuple(
            f"{source_id}-cap-edge-{first}-{second}" for first, second in segments
        ),
        segment_occurrence_paths=((),) * len(segments),
        region_occurrence_paths=((),),
        boundary="fixed",
    )
    source = NativePlcSource(complex_, source_id, source_revision)
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    inside = np.flatnonzero(np.asarray(surface.cover.value_upper) < 0.0)
    if not inside.size:
        raise ValueError("The quotient cover has no certified interior leaf to seed.")
    box = surface.cover.boxes[int(inside[0])]
    seed = np.asarray(cell.cartesian(0.5 * box[0] + 0.5 * box[1]), dtype=np.float64)
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            ()
            if maximum_edge_length is None
            else (UniformSizeControl(scope, float(maximum_edge_length)),)
        ),
        region_seeds=(
            RegionSeed(seed, "region", f"{source_id}-material", RegionRole.SOLID),
        ),
    )
    contract = (
        SpatialCoordinateContract.si()
        if coordinate_contract is None
        else coordinate_contract
    )
    lifted = (
        NativeMeshingProvider(
            NativeMeshingOptions(
                "plc_tetrahedral",
                volume_schedule=(
                    NativeVolumeSchedule(shape_strength=SizeControlStrength.SOFT)
                    if volume_schedule is None
                    else volume_schedule
                ),
            )
        )
        .plan(source, specification, coordinate_contract=contract)
        .execute()
    )
    mesh = lifted.mesh
    output = np.asarray(mesh.coordinates, dtype=np.float64)
    source_rows = {coordinates[row].tobytes(): row for row in range(coordinates.shape[0])}
    count = output.shape[0]
    output_reps = np.arange(count, dtype=np.int64)
    output_shifts = np.zeros((count, 3), dtype=np.int64)
    output_of_source = np.full(coordinates.shape[0], -1, dtype=np.int64)
    for row in range(count):
        source_row = source_rows.get(output[row].tobytes())
        if source_row is not None:
            output_of_source[source_row] = row
    if np.any(output_of_source < 0):
        raise ValueError("The fixed-boundary fill dropped a source vertex.")
    for source_row in range(coordinates.shape[0]):
        row = output_of_source[source_row]
        output_reps[row] = output_of_source[representatives[source_row]]
        output_shifts[row] = windings[source_row]
    # The periodic routes and their orbit-locked adaptation owners carry
    # oriented "tetrahedron" blocks; the native fill's simplex rows are kept.
    blocks = tuple(
        CellBlock(
            block.name,
            "tetrahedron",
            np.asarray(block.vertices, dtype=np.int64),
            global_ids=np.asarray(block.global_ids, dtype=np.int64),
        )
        for block in mesh.blocks
    )
    unbound = CellMesh(output, blocks, vertex_global_ids=mesh.vertex_global_ids)
    topology = PeriodicMeshTopology(unbound, cell, output_reps, output_shifts)
    periodic = CellMesh(
        output,
        blocks,
        vertex_global_ids=mesh.vertex_global_ids,
        periodic_topology=topology,
    )
    result = certify_cell_mesh(periodic, contract)
    volumes = []
    for block in periodic.blocks:
        cells = output[np.asarray(block.vertices, dtype=np.int64)]
        edges = cells[:, 1:] - cells[:, :1]
        volumes.append(np.linalg.det(edges) / 6.0)
    return PeriodicImplicitVolume(
        surface,
        result,
        lifted,
        cap_triangle_count=cap_count,
        quotient_region_components=_quotient_components(periodic, topology),
        measure=float(np.sum(np.abs(np.concatenate(volumes)))),
    )
